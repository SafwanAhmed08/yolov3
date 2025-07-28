import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from tqdm import tqdm

# Set your paths
LABEL_FOLDER = "Annotation_cyphertek.v5i.yolov5pytorch/train/labels"
IMAGE_FOLDER = "Annotation_cyphertek.v5i.yolov5pytorch/train/images"
N_CLUSTERS = 9  # YOLOv3 uses 9 anchors

def load_dataset():
    data = []
    for label_file in tqdm(os.listdir(LABEL_FOLDER)):
        if not label_file.endswith('.txt'):
            continue
        img_file = label_file.replace('.txt', '.jpg')
        img_path = os.path.join(IMAGE_FOLDER, img_file)
        label_path = os.path.join(LABEL_FOLDER, label_file)

        if not os.path.exists(img_path):
            img_path = img_path.replace('.jpg', '.png')  # Try .png fallback
            if not os.path.exists(img_path):
                continue

        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]

        with open(label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                _, x, y, bw, bh = map(float, parts)
                abs_w = bw * w
                abs_h = bh * h
                data.append([abs_w, abs_h])

    return np.array(data)

def generate_anchors(data, n_clusters=9):
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(data)
    anchors = kmeans.cluster_centers_
    anchors = sorted(anchors, key=lambda x: x[0] * x[1])  # sort by area
    return anchors

if __name__ == "__main__":
    print("🚀 Loading dataset...")
    data = load_dataset()
    print(f"✅ Loaded {len(data)} bounding boxes")

    print("📦 Running KMeans clustering...")
    anchors = generate_anchors(data)

    formatted = ",".join([f"{int(w)},{int(h)}" for w, h in anchors])
    print("\n🔧 Paste this into your YOLOv3 .cfg file:")
    print(f"anchors = {formatted}")