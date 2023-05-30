from tqdm import tqdm
import pandas as pd
import os
import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader

labels = next(os.walk('/content/256x256/sketch/tx_000000001110'))[1]

clip_labels = [f"A photo of a {name}." for name in labels]

model_id = "openai/clip-vit-large-patch14"
processor = CLIPProcessor.from_pretrained(model_id)
model = CLIPModel.from_pretrained(model_id)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# create label tokens
label_tokens = processor(
    text=clip_labels,
    padding=True,
    images=None,
    return_tensors='pt'
).to(device)

# encode tokens to sentence embeddings
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

df = pd.read_csv("stats.csv")
categories = np.array(df["Category"].tolist())

class SketchyDataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(df["Category"])

    def __getitem__(self, idx):
        img_id = df["ImageNetID"][idx]
        truth_label = df["Category"][idx]
        label = truth_label.split(" ")
        label = ("_").join(label)
        sketchid = df["SketchID"][idx]
        img_path = os.path.join(self.img_dir, label, img_id + "-" + str(sketchid) + ".png")
        image = default_loader(img_path)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, truth_label

ds = SketchyDataset(img_dir="/content/256x256/sketch/tx_000000001110")

def my_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    return [data, target]

dataloader = DataLoader(
    ds, batch_size=32, shuffle=False, collate_fn=my_collate)

class_probs = {}
for i, (img_list, label_list) in enumerate(tqdm(dataloader)):
    pred_labels = np.array(labels)[classify_images(img_list)]

    for i, label in enumerate(label_list):
         if label not in class_probs:
            class_probs[label] = {'correct': 0,
                                  'total': 0, 'w_labels': [], 'c_labels': []}
            class_probs[label]['total'] += 1
         if pred_labels[i] == label_list[i]:
            class_probs[label]['correct'] += 1
            class_probs[label]['c_labels'].append(pred_labels[i])
         else:
            class_probs[label]['w_labels'].append(pred_labels[i])

class_acc = {}
total_correct = 0
total = 0
for key, value in class_probs.items():
  class_acc[key] = value['correct']/value['total']
  total_correct += value['correct']
  total += value['total']

print("\n Accuracy: " + str(total_correct/total*100))