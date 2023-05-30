from tqdm import tqdm
import os
import cairo
import gi
gi.require_version("Gtk", "3.0")
import numpy as np
import torch
from PIL import Image
import bezier
import clip

model, clip_preprocess = clip.load("ViT-L/14")
model.to("cuda")
model.eval()

def load_hierarchy(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        hierarchy_pairs = [tuple(map(int, line.split())) for line in lines]
    return dict(hierarchy_pairs)

hierarchy_dict = load_hierarchy('../nabirds_info/hierarchy.txt')

def get_hierarchy(label):
    hierarchy_list = [label]
    level = 0
    while label in hierarchy_dict and level < 2:
        label = hierarchy_dict[label]
        hierarchy_list.append(label)
        level += 1
    return hierarchy_list

species_classes = [label for label, value in hierarchy_dict.items() if value != 0 and label not in hierarchy_dict.values()]

def load_class_names(dataset_path=''):
  names = {}
  with open(os.path.join(dataset_path, 'classes.txt')) as f:
    for line in f:
      pieces = line.strip().split()
      class_id = pieces[0]
      names[class_id] = ' '.join(pieces[1:])
  return names

def get_labels():
    labels = []
    class_names = load_class_names("../nabirds_info/")
    for _, name in class_names.items():
        labels.append(f"{name}")
    return labels


class_names = load_class_names("../nabirds_info/")

def classify_image(img, label_emb):
    processed_img = clip_preprocess(img[0]).unsqueeze(0).to("cuda")
    with torch.no_grad():
      img_emb = model.encode_image(processed_img)
      img_emb /= img_emb.norm(dim=-1, keepdim=True)
    
    img_emb = img_emb.detach().cpu().numpy()
    score = np.dot(img_emb, label_emb.T)[0][0]
    return score


def abstract_img(shapes, id, label_emb, idx):
    strokes = []
    for s in shapes:
        strokes.append(s.points.cpu().detach().numpy())

    stroke_lens = []
    for stroke in strokes:
        nodes = np.asfortranarray(stroke.T)
        curve = bezier.Curve(nodes, degree=3)
        stroke_lens.append(curve.length)

    sorted_lens_idx = np.argsort(stroke_lens)
    strokes = [strokes[i] for i in sorted_lens_idx]

    np.save('strokes.npy', strokes)

    def generate_png(strokes, img_name):
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 224, 224)
        ctx = cairo.Context(surface)
        ctx.set_line_width(3)
        ctx.set_source_rgb(0, 0, 0)

        for i, stroke in enumerate(strokes):
            ctx.move_to(stroke[0][0], stroke[0][1])
            for i in range(1, len(stroke), 3):
                ctx.curve_to(stroke[i][0], stroke[i][1], stroke[i+1]
                             [0], stroke[i+1][1], stroke[i+2][0], stroke[i+2][1])
            ctx.stroke()

        surface.write_to_png(img_name)

    generate_png(strokes, id + "_original.png")

    def on_abstraction(strokes, id):
        best_strokes = []
        img = Image.open(id + "_original.png")
        original_img = Image.new("RGBA", img.size, "WHITE")
        original_img.paste(img, (0, 0), img)
        original_img.convert("RGB")
        best_score = classify_image([original_img], label_emb)
        strokes_len = len(strokes)
        pop_num = 0
        total_removed = 0
        for i in range(0, strokes_len):
            np.save(id + "_temp_strokes.npy", strokes)
            strokes.pop(pop_num)
            generate_png(strokes, id + "_temp.png")
            new_img = Image.open(id + "_temp.png")
            tmp_img = Image.new("RGBA", new_img.size, "WHITE")
            tmp_img.paste(new_img, (0, 0), new_img)
            tmp_img.convert("RGB")
            score = classify_image([tmp_img], label_emb)
            cs_diff = best_score - score
            if cs_diff <= 0:
                best_score = score
                best_strokes.append(i)
                total_removed += 1
            else:
                strokes = np.load(id + "_temp_strokes.npy",
                                  allow_pickle=True).tolist()
                pop_num += 1

        generate_png(strokes, "" + id + "_abstracted_img_" + str(idx) + ".png")
        return total_removed
    
    total_removed = on_abstraction(strokes, id)
    return total_removed

import pydiffvg
def load_svg(path_svg):
    svg = os.path.join(path_svg)
    canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(
        svg)
    return canvas_width, canvas_height, shapes, shape_groups

test_imgs = np.load("test_imgs_new.npy", allow_pickle=True)

def run_abstraction(path, num_strokes, seed, on_text = True):
    for im in tqdm(test_imgs):
        removed = []
        for i in [1200, 1220, 1240, 1260, 1280, 1300, 1320, 1340, 1360, 1380]:
            im_split = im.split("/")
            file_name = im_split[-1]
            im_name = file_name.split(".")[0]
            truth_label = int(im_split[-2])
            iter_path = f"{path}{im_name}/{im_name}_{num_strokes}strokes_seed{seed}/svg_logs/svg_iter{i}.svg"
            
            _, _, shapes, _ = load_svg(iter_path)
            species_name = class_names[str(get_hierarchy(truth_label)[-2])]
            input_img = Image.open(f"{path}{im_name}/{im_name}_{num_strokes}strokes_seed{seed}/input.png").convert("RGB")


            if on_text:
                tokens = clip.tokenize(["A grayscale photo of a " + species_name + "."]).to("cuda")
                with torch.no_grad():
                    label_emb = model.encode_text(tokens)
                    label_emb /= label_emb.norm(dim=-1, keepdim=True)
                label_emb = label_emb.detach().cpu().numpy()
                removed.append(abstract_img(shapes, im_name, label_emb, i))

            else:
                processed_img = clip_preprocess(input_img).unsqueeze(0).to("cuda")
                with torch.no_grad():
                    img_emb = model.encode_image(processed_img)
                    img_emb /= img_emb.norm(dim=-1, keepdim=True)
            
                img_emb = img_emb.detach().cpu().numpy()
                removed.append(abstract_img(shapes, im_name, img_emb, i))

        totally_removed.append(np.mean(removed)), removed
    return np.mean(totally_removed), removed

totally_removed, removed = run_abstraction("./92_64s_bw/", 64, 2000, on_text=True)

print(np.mean(totally_removed))
np.save("total_removed.npy", removed)
