from tqdm import tqdm
import matplotlib.pyplot as plt
from cairosvg import svg2png
from io import BytesIO
import pandas as pd
import os
import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
from torchvision.datasets.folder import default_loader

labels = next(os.walk('png'))[1]
labels = np.sort(labels)
new_labels = [item for item in labels for i in range(80)]
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

class HDHSODataset(Dataset):
    def __init__(self, img_dir, transform=None, target_transform=None):
        self.img_labels = new_labels
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels[idx], str(idx + 1) + ".png")
        image = default_loader(img_path)
        label = self.img_labels[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


ds = HDHSODataset(img_dir="/content/png")

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