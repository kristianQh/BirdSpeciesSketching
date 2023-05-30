import clip
import torchvision.transforms as transforms
import random
import torch
import numpy as np
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
from io import BytesIO
from cairosvg import svg2png
from sklearn.metrics import cohen_kappa_score

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
    'A photo of a {}, a type of bird.'
]

def zeroshot_classifier(classnames, templates):
    with torch.no_grad():
        zeroshot_weights = []
        for classname in tqdm(classnames):
            texts = [template.format(classname)
                     for template in templates]  # format with class
            texts = clip.tokenize(texts).cuda()
            class_embeddings = model.encode_text(
                texts)  # embed with text encoder
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

def run_predictions(test_imgs, evalset_path, num_strokes, iters):
    label_names = []
    im_names = []
    class_probs = {}

    for im in tqdm(test_imgs):
        im_split = im.split("/")
        file_name = im_split[-1]
        im_name = file_name.split(".")[0]
        truth_label = int(im_split[-2])

        paths = [
            f"{evalset_path}/{im_name}/{im_name}_{num_strokes}strokes_seed0/svg_logs/svg_iter{i}.svg" for i in iters]
        pngs = [svg2png(url=p) for p in paths]
        imgs = [Image.open(BytesIO(png)) for png in pngs]
        new_imgs = [Image.new("RGB", img.size, "WHITE") for img in imgs]
        for i, img in enumerate(imgs):
            new_imgs[i].paste(img, (0, 0), img)

        pred = classify_images(new_imgs)

        label_names.append([class_names[str(get_hierarchy(species_inds[pred])[-2])]])

    return label_names

def check_stability(test_imgs, start_iter, end_iter, interval, evalset_path, num_strokes, num_ensembles):
    ensemble_iters = [i for i in range(start_iter, end_iter, interval)]
    random.shuffle(ensemble_iters)
    iters1 = ensemble_iters[:num_ensembles]
    print(iters1)
    iters2 = ensemble_iters[num_ensembles:num_ensembles*2]
    print(iters2)

    lbls1 = run_predictions(test_imgs, evalset_path, num_strokes, iters1)
    lbls2 = run_predictions(test_imgs, evalset_path, num_strokes, iters2)

    eq_predictions = 0
    for i in range(len(lbls1)):
        if lbls1[i] == lbls2[i]:
            eq_predictions += 1

    print(f"Equal predictions: {eq_predictions}")
    print(f"Matching rate: " + str(eq_predictions/len(lbls1) * 100))
    print(f"Cohen's Kappa: {cohen_kappa_score(lbls1, lbls2)}")

check_stability(test_imgs, 600, 1400, 20, "70s_colored_92sketches", 70, 10)
