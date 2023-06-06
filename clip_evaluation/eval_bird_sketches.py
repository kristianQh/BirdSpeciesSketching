import numpy as np
import torch
import torchvision.transforms as transforms
import os
import clip
import random
import pydiffvg
import bezier
from PIL import Image
from tqdm import tqdm
from cairosvg import svg2png
from io import BytesIO

def load_strokes(svg_path):
    strokes = []
    _, _, shapes, _ = pydiffvg.svg_to_scene(svg_path)
    for s in shapes:
        if s.stroke_width.data.cpu() <= 0.1:
            continue
        strokes.append(s.points.cpu().detach().numpy())

    if len(strokes) <= 1:
        print("No curves found in the svg file")
    # Load curve points and strokes with bezier
    curve_lens = []
    for i, stroke in enumerate(strokes):
        x, y, x1, y1 = stroke[0][0], stroke[0][1], stroke[1][0], stroke[1][1]
        x2, y2, x3, y3 = stroke[2][0], stroke[2][1], stroke[3][0], stroke[3][1]
        nodes = np.asfortranarray([[x, x1, x2, x3], [y, y1, y2, y3]])
        nodes[1, :] *= -1
        curve = bezier.Curve(nodes, degree=3)
        curve_lens.append(curve.length)

    return np.mean(curve_lens)


def load_hierarchy(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        hierarchy_pairs = [tuple(map(int, line.split())) for line in lines]
    return dict(hierarchy_pairs)

hierarchy_dict = load_hierarchy('nabirds_info/hierarchy.txt')

def get_hierarchy(label):
    hierarchy_list = [label]
    level = 0
    while label in hierarchy_dict and level < 2:
        label = hierarchy_dict[label]
        hierarchy_list.append(label)
        level += 1
    return hierarchy_list

species_inds = [label for label, value in hierarchy_dict.items() if value != 0 and label not in hierarchy_dict.values()]

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
    class_names = load_class_names("nabirds_info/")
    for _, name in class_names.items():
        labels.append(f"{name}")
    return labels


class_names = load_class_names("nabirds_info/")
ALL_LABELS = get_labels()
SPECIES_LABELS = np.array(ALL_LABELS)[species_inds].tolist()

model, clip_preprocess = clip.load("ViT-L/14")
model.eval()

ensemble_prompts = [
    'A photo of a {}, a type of bird.',
]

def zeroshot_classifier(classnames, templates):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template.format(classname)
                     for template in templates]  # format with class
            texts = clip.tokenize(texts).cuda()
            class_embeddings = model.encode_text(
                texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            zeroshot_weights.append(class_embeddings)
    zeroshot_weights = torch.cat(zeroshot_weights, dim=0)
    return zeroshot_weights

zeroshot_weights = zeroshot_classifier(SPECIES_LABELS, ensemble_prompts)

transform = transforms.Compose([transforms.ToTensor()])
normalize_transform = transforms.Compose([
    clip_preprocess.transforms[0],  # Resize
    clip_preprocess.transforms[1],  # CenterCrop
    clip_preprocess.transforms[-1],  # Normalize
])

def classify_images(img_list):
    image_tensors = []
    for img in img_list:
        image_tensors.append(clip_preprocess(img))

    num_imgs = len(image_tensors)
    images_tensor = torch.stack(image_tensors, 0).cuda()

    with torch.no_grad():
      img_emb = model.encode_image(images_tensor)
      img_emb /= img_emb.norm(dim=-1, keepdim=True)

    logits = 100.0 * img_emb @ zeroshot_weights.T
    logits = logits.view(num_imgs, len(SPECIES_LABELS), len(ensemble_prompts))
    average_logits = logits.mean(dim=0)
    average_logits = average_logits.mean(dim=1)
    pred = torch.argmax(average_logits)

    return pred.item()


test_imgs = np.load("nabirds_info/eval_imgs.npy", allow_pickle=True)
def run_predictions(test_imgs, evalset_path, num_strokes, iters, seed, compute_curve_lens=False):
    label_names = []
    im_names = []
    class_probs = {}
    avg_strokes = []

    species_correct = 0
    family_correct = 0
    curve_lens = []

    for im in tqdm(test_imgs):

        im_split = im.split("/")
        file_name = im_split[-1]
        im_name = file_name.split(".")[0]
        truth_label = int(im_split[-2])
        paths = [
            f"{evalset_path}/{im_name}/{im_name}_{num_strokes}strokes_seed{seed}/svg_logs/svg_iter{i}.svg" for i in iters]
        

        if compute_curve_lens:
            bird_curve_lens = []
            for svg in paths:
                crv_lens = load_strokes(svg)
                bird_curve_lens.append(crv_lens)
            curve_lens.append(bird_curve_lens)

            sketch_strokes = np.load(f"{evalset_path}/{im_name}/{im_name}_{num_strokes}strokes_seed{seed}/num_strokes.npy")
            avg_strokes.append(sketch_strokes)
        else:
            avg_strokes.append(0)
            curve_lens.append(0)

        pngs = [svg2png(url=p) for p in paths]
        imgs = [Image.open(BytesIO(png)) for png in pngs]

        new_imgs = [Image.new("RGB", img.size, "WHITE") for img in imgs]
        for i, img in enumerate(imgs):
            new_imgs[i].paste(img, (0, 0), img)

        pred = classify_images(new_imgs)

        # Get species accuracy
        class_probs[im_name] = class_names[str(get_hierarchy(species_inds[pred])[-2])]
        if get_hierarchy(species_inds[pred])[-2] == get_hierarchy(truth_label)[-2]:
            species_correct += 1
            im_names.append(im_name)

        # Get family accuracy
        if get_hierarchy(species_inds[pred])[-1] == get_hierarchy(truth_label)[-1]:
            family_correct += 1


        label_names.append([class_names[str(get_hierarchy(species_inds[pred])[-2])]])

    return label_names, im_names, species_correct/len(test_imgs)*100, family_correct/len(test_imgs)*100,  np.mean(curve_lens), np.mean(avg_strokes)

def compare_accuricies(test_imgs, start_iter, end_iter, interval, evalset_path1, evalset_path2, num_strokes, num_ensembles, seed):
    ensemble_iters = [i for i in range(start_iter, end_iter, interval)]
    random.shuffle(ensemble_iters)
    iters = ensemble_iters[:num_ensembles]
    _, _, species_acc1, fam_acc1, avg_curve_lens1, avg_strokes1 = run_predictions(
        test_imgs, evalset_path1, num_strokes, iters, seed)
    _, _, species_acc2, fam_acc2, avg_curve_lens2, avg_strokes2 = run_predictions(
        test_imgs, evalset_path2, num_strokes, iters, seed)

    print("Species accuracy 1:", species_acc1)
    print("Familly accuracy 1:", fam_acc1)
    print("---------")
    print("Species accuracy 2:", species_acc2)
    print("Familly accuracy 2:", fam_acc2)    


compare_accuricies(test_imgs, 1100, 1300, 20, "70s_colored_92sketches", "70s_colored_92sketches", 70, 10, 0)
