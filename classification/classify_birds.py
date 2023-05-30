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
# model_id = 'openai/clip-vit-large-patch14'
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
        labels.append(f"This is a photo of {name}.")
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


def my_collate(batch):
    data = [item[0] for item in batch]
    # target = [hier_tree[item[1]][-1] for item in batch]
    target = [item[1] for item in batch]
    path = [item[2] for item in batch]
    return [data, np.array(target), path]


train_dataloader = DataLoader(
    train_dataset, batch_size=32, shuffle=False, collate_fn=my_collate)
test_dataloader = DataLoader(
    test_dataset, batch_size=32, shuffle=False, collate_fn=my_collate)

class_probs = {}
for i, (img_list, label_list, path) in enumerate(tqdm(test_dataloader)):
  preds = classify_images(img_list)
  preds = np.array(hier_2)[preds]

  for i, label in enumerate(label_list):
    if label not in class_probs:
      class_probs[label] = {'correct': 0, 'total': 0, 'c_pred_imgs': [
      ], 'w_pred_imgs': [], 'w_labels': [], 'c_labels': []}
    class_probs[label]['total'] += 1
    if hier_tree[preds[i]][-2] == hier_tree[label_list[i]][-2]:
      class_probs[label]['correct'] += 1
      class_probs[label]['c_pred_imgs'].append(path[i])
      class_probs[label]['c_labels'].append(
          class_names[str(hier_tree[preds[i]][-1])])
    else:
      class_probs[label]['w_pred_imgs'].append(path[i])
      class_probs[label]['w_labels'].append(
          class_names[str(hier_tree[preds[i]][-1])])

class_acc = {}
total_correct = 0
total = 0
for key, value in class_probs.items():
  class_acc[key] = value['correct']/value['total']
  total_correct += value['correct']
  total += value['total']

print("\n Accuracy: " + str(total_correct/total*100))

# np.save('new/class_probs_u2net_eval.npy', class_probs)

# zipped = list(class_acc.items())
# res = sorted(zipped, key=lambda x: x[1])
# np.save('new/class_acc_u2net_eval.npy', res)