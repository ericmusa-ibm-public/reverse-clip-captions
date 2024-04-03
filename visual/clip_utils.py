import numpy as np
import open_clip
from PIL import Image
import torch

model, _, preprocess = open_clip.create_model_and_transforms('convnext_base_w', pretrained='laion2b_s13b_b82k_augreg')
# model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='commonpool_m_clip_s128m_b4k')

model.eval()
context_length = model.context_length
vocab_size = model.vocab_size

print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in model.parameters()]):,}")
print("Context length:", context_length)
print("Vocab size:", vocab_size)


def process_images(images, features=True):
    result = torch.tensor(np.stack([preprocess(_) for _ in images]))
    if features:
        with torch.no_grad():
            result = model.encode_image(result).float()
            result /= result.norm(dim=-1, keepdim=True)
    return result


def process_texts(texts, features=True):
    result = open_clip.tokenizer.tokenize(texts)
    if features:
        with torch.no_grad():
            result = model.encode_text(result).float()
            result /= result.norm(dim=-1, keepdim=True)
    return result


def feature_similarities(features_a, features_b):
    return features_a.cpu().numpy() @ features_b.cpu().numpy().T


def texts_images_similarities(texts, images):
    if all(isinstance(_, str) for _ in images):
        images = [Image.open(_).convert('RGB') for _ in images]
    text_features = process_texts(texts)
    image_features = process_images(images)
    return feature_similarities(text_features, image_features)


import matplotlib.pyplot as plt

def similarities_heat_map(texts, images):
    similarities = texts_images_similarities(texts, images)
    plt.figure(figsize=(20, 14))
    plt.imshow(similarities, vmin=0.1, vmax=0.5)
    plt.colorbar()
    plt.yticks(range(len(texts)), texts, fontsize=18)
    plt.xticks([])
    for i, image in enumerate(images):
        plt.imshow(image, extent=(i - 0.5, i + 0.5, -1.6, -0.6), origin="lower")
    for x in range(similarities.shape[1]):
        for y in range(similarities.shape[0]):
            plt.text(x, y, f"{similarities[y, x]:.2f}", ha="center", va="center", size=12)

    for side in ["left", "top", "right", "bottom"]:
        plt.gca().spines[side].set_visible(False)

    plt.xlim([-0.5, len(images) - 0.5])
    plt.ylim([len(texts) + 0.5, -2])

    plt.title("Cosine similarity between text and image features", size=20)


if __name__ == "__main__":
    import os

    OCR_TESTING = True

    if OCR_TESTING:
        image_root = os.path.join('images', 'OCR testing')
    else:
        image_root = 'images'

    image_paths = [os.path.join(image_root, _) for _ in os.listdir(image_root) if _.endswith('.png')]
    images_dict = {_: Image.open(_).convert("RGB") for _ in image_paths}

    images = []
    texts = []

    for fname, image in images_dict.items():
        images.append(image)
        if not OCR_TESTING:
            texts.append(os.path.split(fname)[1].split('.')[0])

    if OCR_TESTING:
        texts += \
            ['red', 'circle'] + \
            ['turtle', 'octagon', 'triangle']  # + \
            # ['quantum mechanics', '40%', 'percentage']  # + \
            # ['quantum mechanics', 'gown', '40%', 'percentage', '30%', '5 L', 'liters'] + \
    similarities_heat_map(texts, images)
    plt.show()