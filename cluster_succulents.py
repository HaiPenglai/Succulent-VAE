import os
import cv2
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
import umap
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import shutil
from tqdm import tqdm

# --- 配置 ---
INPUT_DIR = './dataset'
OUTPUT_DIR = './target_classified_dataset_15'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_CLASSES = 15  # 你想要的类别数量
BATCH_SIZE = 32


# 1. 强化色彩特征提取 (特别增强 Hue 色调)
def get_strong_color_features(img_path):
    img = cv2.imread(img_path)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    # 掩码：只关注多肉，不关注黑色背景
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)[1]
    
    # 我们把色调(H)分得更细(32个区间)，饱和度(S)和亮度(V)分得粗一点
    hist_h = cv2.calcHist([img_hsv], [0], mask, [32], [0, 180])
    hist_s = cv2.calcHist([img_hsv], [1], mask, [8], [0, 256])
    
    # 归一化
    hist_h = cv2.normalize(hist_h, hist_h).flatten()
    hist_s = cv2.normalize(hist_s, hist_s).flatten()
    
    # 给 H 分配更高的权重 (乘以3)，让颜色成为主导因素
    return np.concatenate([hist_h * 3.0, hist_s])

# 2. 模型加载
print("正在加载模型并提取特征...")
model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(DEVICE)
model.eval()

class SimpleImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.image_files = sorted([f for f in os.listdir(root_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        self.transform = transform
    def __len__(self): return len(self.image_files)
    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        if self.transform: image = self.transform(image)
        return image, img_name

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dataset = SimpleImageDataset(root_dir=INPUT_DIR, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# 3. 特征采集
deep_features = []
color_features = []
filenames = []

with torch.no_grad():
    for imgs, names in tqdm(dataloader):
        imgs = imgs.to(DEVICE)
        feat = model(imgs).cpu().numpy()
        deep_features.append(feat)
        filenames.extend(names)
        for name in names:
            c_feat = get_strong_color_features(os.path.join(INPUT_DIR, name))
            color_features.append(c_feat)

deep_features = np.concatenate(deep_features, axis=0)
color_features = np.array(color_features)

# 4. 特征融合与加权
# 标准化深度特征
scaler = StandardScaler()
deep_features = scaler.fit_transform(deep_features)

# 再次强化颜色特征的绝对影响力
# 这里的 5.0 是颜色相对于形状的杠杆率
combined_features = np.hstack([deep_features * 1.0, color_features * 8.0])

# 5. UMAP 降维 (保持全局结构)
print(f"正在降维并合并为 {TARGET_CLASSES} 类...")
reducer = umap.UMAP(
    n_neighbors=30,      # 增大这个值，会让算法更倾向于“大类合并”
    min_dist=0.1,
    n_components=20,     # 保留足够的信息用于聚类
    metric='cosine',
    random_state=42
)
embedding = reducer.fit_transform(combined_features)

# 6. 使用 Agglomerative Clustering 强行指定类别数
# 这种算法会不断合并最接近的样本，直到剩下 TARGET_CLASSES 个类
clusterer = AgglomerativeClustering(
    n_clusters=TARGET_CLASSES, 
    linkage='ward'  # ward 能够很好地处理大小不一的簇
)
labels = clusterer.fit_predict(embedding)

# 7. 导出
if os.path.exists(OUTPUT_DIR): shutil.rmtree(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR)

counts = {}
for filename, label in zip(filenames, labels):
    label_str = f"class_{label:03d}"
    target_folder = os.path.join(OUTPUT_DIR, label_str)
    if not os.path.exists(target_folder): os.makedirs(target_folder)
    shutil.copy(os.path.join(INPUT_DIR, filename), os.path.join(target_folder, filename))
    counts[label_str] = counts.get(label_str, 0) + 1

print("\n--- 分类完成 (强行聚类模式) ---")
# 打印排序后的分类结果
for cls in sorted(counts.keys()):
    print(f"类别 {cls}: {counts[cls]} 张图片")