import os
import numpy as np
import torch
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from torch.utils.data import DataLoader
import os
import numpy as np
from nabirds import NABirds

def load_hierarchy(filename):
    with open(filename, 'r') as f:
        lines = f.readlines()
        hierarchy_pairs = [tuple(map(int, line.split())) for line in lines]
    return dict(hierarchy_pairs)

def get_hierarchy(label, hierarchy):
    hierarchy_list = [label]
    level = 0
    while label in hierarchy and level < 2:
        label = hierarchy[label]
        hierarchy_list.append(label)
        level += 1
    return hierarchy_list

hierarchy_dict = load_hierarchy('nabirds_info/hierarchy.txt')
species_classes = [label for label, value in hierarchy_dict.items() if value != 0 and label not in hierarchy_dict.values()]

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

def get_labels():
    labels = []
    class_names = load_class_names("nabirds_info/")
    for _, name in class_names.items():
        labels.append(f"{name}")
    return labels

class_names = load_class_names("nabirds_info/")
ALL_LABELS = get_labels()
SPECIES_LABELS = np.array(ALL_LABELS)[species_classes].tolist()
# create label tokens
label_tokens = processor(
    text=SPECIES_LABELS,
    padding=True,
    images=None,
    return_tensors='pt'
).to(device)

with torch.no_grad():
  label_emb = model.get_text_features(**label_tokens)
label_emb /= label_emb.norm(dim=-1, keepdim=True)
label_emb = label_emb.detach().cpu().numpy()

def classify_images(img_list):
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
    scores = img_emb @ label_emb.T
    pred = np.argmax(scores, axis=1)
    return pred

def my_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    path = [item[2] for item in batch]
    return [data, np.array(target), path]

train_dataloader = DataLoader(
    train_dataset, batch_size=1, shuffle=False, collate_fn=my_collate)
test_dataloader = DataLoader(
    test_dataset, batch_size=1, shuffle=False, collate_fn=my_collate)

class_probs = {}
for i, (img_list, label_list, path) in enumerate(tqdm(test_dataloader)):
  preds = classify_images(img_list)
  preds = np.array(species_classes)[preds]

  for i, label in enumerate(label_list):
    if label not in class_probs:
      class_probs[label] = {'correct': 0, 'total': 0, 'c_pred_imgs': [
      ], 'w_pred_imgs': [], 'w_labels': [], 'c_labels': []}
    class_probs[label]['total'] += 1
    if get_hierarchy(preds[i])[-2] == get_hierarchy(label_list[i])[-2]:
      class_probs[label]['correct'] += 1
      class_probs[label]['c_pred_imgs'].append(path[i])
      class_probs[label]['c_labels'].append(
          class_names[str(get_hierarchy(preds[i])[-2])])
    else:
      class_probs[label]['w_pred_imgs'].append(path[i])
      class_probs[label]['w_labels'].append(
          class_names[str(get_hierarchy(preds[i])[-2])])

class_acc = {}
total_correct = 0
total = 0
for key, value in class_probs.items():
  class_acc[key] = value['correct']/value['total']
  total_correct += value['correct']
  total += value['total']

print("\n Accuracy: " + str(total_correct/total*100))
