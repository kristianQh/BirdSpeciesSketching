import math
import numpy as np
import bezier
import matplotlib.pyplot as plt
from scipy import interpolate
import pydiffvg

def plot_sketch(curves, sketch_name):
    fig, ax = plt.subplots()
    plt.rcParams['lines.linewidth'] = 2
    for b in curves:
        b.plot(num_pts=256, ax=ax, color='black')

    ax.axis('scaled')
    plt.axis('off')
    plt.savefig(sketch_name + ".svg", bbox_inches='tight')
        

def load_strokes(svg_path):
    strokes = []
    _, _, shapes, _ = pydiffvg.svg_to_scene(svg_path)
    for s in shapes:
        strokes.append(s.points.cpu().detach().numpy())

    # Load curve points and strokes with bezier
    bz_curves = []
    curve_points = []
    for i, stroke in enumerate(strokes):
        x, y, x1, y1 = stroke[0][0], stroke[0][1], stroke[1][0], stroke[1][1]
        x2, y2, x3, y3 = stroke[2][0], stroke[2][1], stroke[3][0], stroke[3][1]
        nodes = np.asfortranarray([[x, x1, x2, x3], [y, y1, y2, y3]])
        nodes[1, :] *= -1
        s_vals = np.linspace(0.0, 1.0, 200)

        curve = bezier.Curve(nodes, degree=3)
        points = curve.evaluate_multi(s_vals)
        curve_points.append([points])
        bz_curves.append(curve)

    plot_sketch(bz_curves, "original")
    return np.array(curve_points), bz_curves

curve_points, bz_curves = load_strokes("best_iter.svg")
def sweep_line(bz_curves, curve_points, threshold=5):
    endpoints = np.array([])
    for i, c in enumerate(bz_curves):
        _curve_points = curve_points[i][0]
        endpoints = np.append(endpoints, np.array(
            [_curve_points[1][0], _curve_points[1][-1]]))


    endpoints = endpoints.reshape(-1, 2)
    # Sort endpoints by y-coordinate
    flat = endpoints.flatten()
    indices = np.argpartition(flat, len(flat)-1)[:len(flat)-1]
    sorted_inds = indices[np.argsort(flat[indices])]
    rows, cols = np.unravel_index(sorted_inds, endpoints.shape)

    visited = []
    closed = []
    for i in range(len(rows)):
        for v in visited:
                if np.linalg.norm(bz_curves[rows[i]].evaluate(cols[i]) - bz_curves[v[0]].evaluate(v[1])) <= threshold:
                    closed.append(((rows[i], cols[i]), (v[0], v[1])))
        visited.append((rows[i], cols[i]))

    for c in closed:
        if c[0][0] != c[1][0]:
            stroke_num = c[0][0]
            stroke_num2 = c[1][0]
            if c[0][1] == 1:
                point = -1
            else:
                point = 0
            if c[1][1] == 1:
                point2 = -1
            else:
                point2 = 0

        cv = bz_curves[stroke_num].nodes
        cv[0][point] = bz_curves[stroke_num2].nodes[0][point2]
        cv[1][point] = bz_curves[stroke_num2].nodes[1][point2]
        bz_curves[stroke_num] = bezier.Curve.from_nodes(cv)
        curve_points[stroke_num] = bz_curves[stroke_num].evaluate_multi(np.linspace(0.0, 1.0, 200))

    return bz_curves, curve_points

bz_curves, curve_points = sweep_line(bz_curves, curve_points, 5)
plot_sketch(bz_curves, "sweep_line")


def angle_between_points(a, b, c):
    ab = [(b[0] - a[0]), (b[1] - a[1])]
    bc = [(c[0] - b[0]), (c[1] - b[1])]
    dot_product = ab[0] * bc[0] + ab[1] * bc[1]
    ab_length = math.sqrt(ab[0] ** 2 + ab[1] ** 2)
    bc_length = math.sqrt(bc[0] ** 2 + bc[1] ** 2)
    if math.isclose(ab_length*bc_length, 0, abs_tol=1e-4):
        return 0
    else:
        tempo = dot_product / (ab_length * bc_length)
        if math.isclose(tempo, 1.0, abs_tol=1e-5):
            tempo = 1.0
        angle = math.acos(tempo)
    return math.degrees(angle)

def split_curve(points, angle_threshold):
    segments = []
    current_segment = [points[0]]
    angles = []
    for i in range(1, len(points) - 1):
        angle = angle_between_points(points[i-1], points[i], points[i+1])
        angles.append(angle)

    max_angle = max(angles) - 1e-8
    if max_angle > angle_threshold:
        angle_threshold = max_angle
    for i in range(1, len(points) - 1):
        current_segment.append(points[i])
        angle = angle_between_points(points[i-1], points[i], points[i+1])
        if angle > angle_threshold:
            current_segment.append(points[i+1])
            segments.append(current_segment)
            current_segment = [points[i+1]]
        elif i == len(points) - 2:
            current_segment.append(points[i+1])
            segments.append(current_segment)
    return segments


def split_curves(cv_points, angle_threshold=10):
    new_curve_points = []
    old_list_size = len(cv_points)
    while True:
        new_curve_points = []
        for i in range(len(cv_points)):
            cx = cv_points[i][0][0]
            cy = cv_points[i][0][1]
            bzc = [(x, y) for x, y in zip(cx, cy)]
            smooth_segments = split_curve(bzc, angle_threshold)
            if len(smooth_segments) > 1:
                seg1x = [x for x, y in smooth_segments[0]]
                seg1y = [y for x, y in smooth_segments[0]]
                cv1 = [seg1x, seg1y]

                seg2x = [x for x, y in smooth_segments[1]]
                seg2y = [y for x, y in smooth_segments[1]]
                cv2 = [seg2x, seg2y]
                new_curve_points.append([np.array(cv1)])
                new_curve_points.append([np.array(cv2)])
            else:
                new_curve_points.append(cv_points[i])

        cv_points = new_curve_points
        new_list_size = len(cv_points)
        print(new_list_size, old_list_size)
        if new_list_size == old_list_size:
            break
        else:
            old_list_size = new_list_size

    new_bz_curves = []
    for i in range(len(new_curve_points)):
        x1 = new_curve_points[i][0][0]
        y1 = new_curve_points[i][0][1]
        # Remove very short segments
        if len(x1) <= 30:
            continue
        else:
            new_bz_curves.append(bezier.Curve.from_nodes(np.array([x1, y1])))
    return new_bz_curves, new_curve_points

new_bz_curves, curve_points = split_curves(curve_points, 10)

bz_curves2 = []
curve_points2 = []
for i, crv in enumerate(new_bz_curves):
    s_vals = np.linspace(0.0, 1.0, 200)
    points = crv.evaluate_multi(s_vals)
    curve_points2.append([points])
    bz_curves2.append(crv)

bz_curves = bz_curves2
curve_points = curve_points2


def min_distance(s, t):
    s_t = np.array(s).T
    s_t = np.squeeze(s_t)
    t_t = np.array(t).T
    t_t = np.squeeze(t_t)
    distances = np.linalg.norm(s_t[:, None] - t_t, axis=2)
    i, j = np.unravel_index(np.argmin(distances), distances.shape)
    min_dist = distances[i, j]
    nearest_segment = (i, j)
    return min_dist, nearest_segment

def unit_tangent_vector(points, index):
    if index == len(points)-1:
        vector = points[-1] - points[-2]
    else:
        vector = points[index+1] - points[index]
    tangent = vector / np.linalg.norm(vector)
    return tangent

def angular_difference(s, t, i_star, j_star):
    s = np.array(s).T
    s = s.reshape((s.shape[0], -1))
    t = np.array(t).T
    t = t.reshape((t.shape[0], -1))
    theta_s = unit_tangent_vector(s, i_star)
    theta_t = unit_tangent_vector(t, j_star)
    angle_diff = 1 - np.abs(np.dot(theta_s, theta_t))

    return angle_diff

def is_curve_in_list(curve, curve_list):
    for existing_curve in curve_list:
        if np.array_equal(curve, existing_curve):
            return True
    return False

def approx_curves(bz_curves, curve_points, dist_thresh=0.1, angle_thresh=0.01):
    stroke_pairs = []
    for i in range(len(bz_curves)):
        for j in range(len(bz_curves)):
            if i == j:
                continue
            else:
                min_dist, nearest_segment = min_distance(curve_points[i], curve_points[j])
                i_star, j_star = nearest_segment
                angle = angular_difference(curve_points[i], curve_points[j], i_star, j_star)
                stroke_pairs.append([i, j, min_dist, angle])

    stroke_pairs_sorted = sorted(stroke_pairs, key=lambda x: (x[2], x[3]))

    interpolated_curves = {i: curve_points[i] for i in range(len(curve_points))}
    equivalent_curves = {i: {i} for i in range(len(curve_points))}

    for pairs in stroke_pairs_sorted:
        cv1_idx, cv2_idx = pairs[0], pairs[1]
        dist, nearest_segment = min_distance(interpolated_curves[cv1_idx], interpolated_curves[cv2_idx])
        angle = angular_difference(interpolated_curves[cv1_idx], interpolated_curves[cv2_idx], nearest_segment[0], nearest_segment[1])
        if dist < dist_thresh:
            if angle <= angle_thresh:
                c1 = np.squeeze(np.array(interpolated_curves[cv1_idx]).T)
                c2 = np.squeeze(np.array(interpolated_curves[cv2_idx]).T)
                new_curve = np.concatenate((c1, c2))
                x,y = new_curve.T
                f=interpolate.interp1d(x, y)
                x_new = np.linspace(x.min(), x.max(), 100)
                y_new = f(x_new)
                bz_curve = bezier.Curve.from_nodes(np.asfortranarray([x_new, y_new]))
                new_curve_points = [bz_curve.evaluate_multi(np.linspace(0, 1, 200))]

                for idx in equivalent_curves[cv1_idx] | equivalent_curves[cv2_idx]:
                    interpolated_curves[idx] = new_curve_points
                equivalent_curves[cv1_idx] |= equivalent_curves[cv1_idx] | equivalent_curves[cv2_idx]

    unique_curves = []
    for curve in interpolated_curves.values():
        if not is_curve_in_list(curve, unique_curves):
            unique_curves.append(bezier.Curve.from_nodes(np.asfortranarray([curve[0][0], curve[0][1]])))

    return unique_curves

final_bz_curves = approx_curves(bz_curves, curve_points, 0.7, 0.035)
plot_sketch(final_bz_curves, 'simplified')
