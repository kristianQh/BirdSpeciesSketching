import os
from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt

dataset = "../CUB_200_2011"

def load_class_names(dataset_path=''):
  names = []
  with open(os.path.join(dataset_path, 'classes.txt')) as f:
    for line in f:
      pieces = line.strip().split()[1]
      split = pieces.split(".")
      spiece_name = split[1].replace("_", " ")
      names.append(spiece_name)
  return names

labels = load_class_names(f"{dataset}")
clip_labels = [f"This is a photo of {name}." for name in labels]

from transformers import CLIPProcessor, CLIPModel
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
    print(scores)
    top_5 = np.argsort(scores, axis=1)[:, -5:]
    print(top_5)
    pred = np.argmax(scores, axis=1)
    return pred

img = Image.open("jay2.jpg")
# # Create a white rgba background
# new_img = Image.new("RGBA", img.size, "WHITE")
# # Paste the image on the background. Go to the links given below for details.
# new_img.paste(img, (0, 0), img)
# new_img.convert('RGB')

print(classify_images([img]))
