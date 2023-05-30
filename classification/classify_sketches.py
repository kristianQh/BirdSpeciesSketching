
import os
from PIL import Image
import numpy as np
import torch
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from torch.utils.data import DataLoader
from collections import defaultdict
from nabirds import NABirds
import os
import numpy as np
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# sketches_fine = np.load("experiment_sketches/sketches_fine_abstract_correct_imgs.npy", allow_pickle=True)
# sketches_veryfine = np.load("experiment_sketches/sketches_very_fine.npy", allow_pickle=True)
# contour_fine = np.load("experiment_sketches/contour_fine_correct_imgs.npy", allow_pickle=True)


# print(len(sketches_fixed))
# print(len(sketches_fixed_abstraction1))

# for im in sketches_fine:
#     if im not in sketches_veryfine:
#         im_split = im.split("/")
#         file_name = im_split[-1]
#         im_name = file_name.split(".")[0]
#         print(im_split[-2])
#         path = "sketches_fixed/" + im_name + "/" + im_name + \
#             "_64strokes_seed0/" + "clip_abstraction0.05.svg"
#         # path = "contour_imgs/" + im_name + ".jpg"
#         shutil.copy(path, "experiment_sketches/" + "best_" + im_name + ".svg")

def save_hierarchy_labels(dirname):
    """
    save the labels in txt file to numpy file
    """
    fid = open(os.path.join(dirname, "hierarchy.txt"))
    lines = fid.readlines()
    num_lines = len(lines)
    class_map = np.ndarray((num_lines, 2), dtype=np.uint)

    for i, line in enumerate(lines):
        line = line.strip('\n')
        line_split = line.split(' ')
        class_map[i, 0], class_map[i, 1] = line_split[0], line_split[1]

    return class_map

class_map = save_hierarchy_labels("nabirds_seg_imgs/")

def find_class_hierarchy(class_map):
    children_chain = list(class_map[:, 0])
    parent_chain = list(class_map[:, 1])
    d = defaultdict(set)
    for i in range(class_map.shape[0]):
        d[class_map[i, 1]].add(class_map[i, 0])
    child_classes = set(class_map[:, 0])
    fine_classes = list(child_classes - d.keys())
    hier_tree = {}
    levl_count = []
    for fine_class in fine_classes:
        i = 1
        hier_tree[fine_class] = []
        hier_tree[fine_class].append(fine_class)
        index_1 = children_chain.index(fine_class)
        class_super = parent_chain[index_1]
        while class_super != 0 and i != 3:
            i += 1
            hier_tree[fine_class].append(class_super)
            index = children_chain.index(class_super)
            class_super = parent_chain[index]
        levl_count.append(i)
    return hier_tree, levl_count


hier_tree, levl_count = find_class_hierarchy(class_map)
train_dataset = NABirds('.', train=True, download=None)
test_dataset = NABirds('.', train=False, download=None)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")


def load_class_names(dataset_path=''):
  names = {}
  with open(os.path.join(dataset_path, 'classes.txt')) as f:
    for line in f:
      pieces = line.strip().split()
      class_id = pieces[0]
      names[class_id] = ' '.join(pieces[1:])
  return names


class_names = load_class_names("nabirds_seg_imgs/")


def get_labels():
    labels = []
    class_names = load_class_names("nabirds_seg_imgs/")
    for _, name in class_names.items():
        labels.append(f"a black and white photo of {name}.")
    return labels


hier_2 = []
for key, value in hier_tree.items():
    hier_2.append(key)


ALL_LABELS = get_labels()
HIER_LABELS = np.array(ALL_LABELS)[hier_2].tolist()
# create label tokens
label_tokens = processor(
    text=HIER_LABELS,
    padding=True,
    images=None,
    return_tensors='pt'
).to(device)

# encode tokens to sentence embeddings
with torch.no_grad():
  label_emb = model.get_text_features(**label_tokens)
# detach from pytorch gradient computation
label_emb /= label_emb.norm(dim=-1, keepdim=True)
label_emb = label_emb.detach().cpu().numpy()


def classify_images(img_list):
    # encode image to image embedding
    with torch.no_grad():
      processed_imgs = processor(
          text=None,
          padding=True,
          images=img_list,
          return_tensors='pt'
      )['pixel_values'].to(device)
      img_emb = model.get_image_features(processed_imgs)
      img_emb /= img_emb.norm(dim=-1, keepdim=True)
    img_emb = img_emb.detach().cpu().numpy()
    scores = np.dot(img_emb, label_emb.T)
    pred = np.argmax(scores, axis=1)
    return pred

import matplotlib.pyplot as plt
import cv2 as cv
from scipy.ndimage import gaussian_filter
from skimage.color import rgb2gray, gray2rgb
from skimage.filters import threshold_otsu
from skimage.util import random_noise
from skimage import feature
import math

def canny(im):
    im = rgb2gray(im)
    im = random_noise(im, mode='speckle', mean=0.1)
    im = feature.canny(im, sigma=2)
    return np.invert(im)

def dog(im):
    gamma = 0.98
    phi = 200
    eps = -0.1
    sigma = 0.8
    im = rgb2gray(im)
    imf1 = gaussian_filter(im, sigma)
    imf2 = gaussian_filter(im, sigma * 3)
    imdiff = imf1 - gamma * imf2
    imdiff = (imdiff < eps) * 1.0  + (imdiff >= eps) * (1.0 + np.tanh(phi * imdiff))
    imdiff -= imdiff.min()

    if imdiff.max() != 0.0:
        imdiff /= imdiff.max()
        th = threshold_otsu(imdiff)
        imdiff = imdiff >= th

    return imdiff

def my_collate(batch):
    data = [Image.fromarray(canny(item[0])).convert("RGB") for item in batch]
    target = [hier_tree[item[1]][-1] for item in batch]
    target = [item[1] for item in batch]
    path = [item[2] for item in batch]
    return [data, np.array(target), path]


train_dataloader = DataLoader(
    train_dataset, batch_size=16, shuffle=False, collate_fn=my_collate)
test_dataloader = DataLoader(
    test_dataset, batch_size=16, shuffle=False, collate_fn=my_collate)

from io import BytesIO
from cairosvg import svg2png

class_probs = {}
test_imgs = np.load("test_imgs.npy", allow_pickle=True)
for im in test_imgs:
    im_split = im.split("/")
    file_name = im_split[-1]
    im_name = file_name.split(".")[0]
    truth_label = int(im_split[-2])
    path = "32-strokes/" + im_name + "/" + im_name + "_32strokes_seed0/" + "best_iter.svg"
    png = svg2png(url=path)
    img = Image.open(BytesIO(png))
    new_img = Image.new("RGBA", img.size, "WHITE")
    new_img.paste(img, (0, 0), img)
    new_img.convert("RGB")
    pred = classify_images([new_img])
    pred = hier_2[pred[0]]
    # print(truth_label)
    # print(hier_tree[truth_label])
    if truth_label not in class_probs:
        class_probs[truth_label] = {'total': 0, 'correct': 0, 'c_pred_imgs': [], 'c_labels': [], 'w_pred_imgs': [], 'w_labels': []}
    class_probs[truth_label]['total'] += 1
    if hier_tree[pred][-2] == hier_tree[truth_label][-2]:
        class_probs[truth_label]['correct'] += 1
        class_probs[truth_label]['c_pred_imgs'].append(im)
        class_probs[truth_label]['c_labels'].append(class_names[str(hier_tree[pred][-2])])
    else:
        class_probs[truth_label]['w_pred_imgs'].append(im)
        class_probs[truth_label]['w_labels'].append(class_names[str(hier_tree[pred][-2])])

class_acc = {}
total_correct = 0
total = 0
for key, value in class_probs.items():
  class_acc[key] = value['correct']/value['total']
  total_correct += value['correct']
  total += value['total']

print("\n Accuracy: " + str(total_correct/total*100))

# correct_imgs = []
# for k in class_probs.keys():
#     if class_probs[k]['c_pred_imgs'] == []:
#         continue
#     else:
#         for i in range(len(class_probs[k]['c_pred_imgs'])):
#             correct_imgs.append(class_probs[k]['c_pred_imgs'][i])
# np.save("experiment_sketches/sketches_very_fine.npy", correct_imgs)
