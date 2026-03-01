import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from segment_anything import sam_model_registry, SamPredictor

# --- 配置参数 ---
INPUT_DIR = './data'
OUTPUT_DIR = './dataset'
CHECKPOINT_PATH = "./sam_vit_h_4b8939.pth"
MODEL_TYPE = "vit_h"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_SIZE = 256

# 初始化 SAM
print(f"正在加载模型...")
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH)
sam.to(device=DEVICE)
predictor = SamPredictor(sam)

def prune_mask(mask, img_w, img_h):
    """
    核心二次处理逻辑（虚拟剪刀）：
    通过 腐蚀 -> 找最大块 -> 膨胀，切断并丢弃边缘连接不紧密的残缺叶片和杂物。
    """
    # ================= 【调参区】 =================
    # 参数1：剪刀大小系数。默认0.025。调到 0.04 或 0.05 力度会大增！
    CUT_FORCE_RATIO = 0.045 
    
    # 参数2：连续下刀次数。默认1。如果杂叶连接很粗，可以改为 2 或 3！
    ITERATIONS = 2          
    # ============================================

    # 1. 计算核大小
    base_size = min(img_w, img_h)
    k_size = max(5, int(base_size * CUT_FORCE_RATIO))
    k_size = k_size if k_size % 2 == 1 else k_size + 1 
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    
    # 2. 腐蚀 (Erosion)：用力切断连接！
    eroded = cv2.erode(mask, kernel, iterations=ITERATIONS)
    
    # 3. 找最大连通块
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(eroded, connectivity=8)
    if num_labels <= 1:
        # 如果你发现原图完全变黑了，说明力度实在太大，把整个多肉都腐蚀没了，此时退回原状
        print("警告：剪裁力度过大，主体被完全消融，退回原掩码。请调小参数！")
        return mask 
        
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    
    pruned_eroded = np.zeros_like(eroded)
    pruned_eroded[labels == largest_label] = 1
    
    # 4. 膨胀 (Dilation)：按相同的次数恢复主体大小！
    pruned_mask = cv2.dilate(pruned_eroded, kernel, iterations=ITERATIONS)
    
    return pruned_mask

def refine_mask(masks, img_w, img_h):
    """基础掩码处理：补全内部洞"""
    areas = [np.sum(m) for m in masks]
    best_mask = masks[np.argmax(areas)].astype(np.uint8)

    # 填补内部空洞 (仅提取最外层轮廓并填充内部)
    contours, _ = cv2.findContours(best_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask_filled = np.zeros_like(best_mask)
    cv2.drawContours(mask_filled, contours, -1, 1, thickness=cv2.FILLED)
    
    return mask_filled

def smooth_edges(mask, img_w, img_h):
    """抗锯齿：生成 Alpha 羽化边缘"""
    mask_float = mask.astype(np.float32)
    blur_size = max(3, int(min(img_w, img_h) * 0.005))
    blur_size = blur_size if blur_size % 2 == 1 else blur_size + 1
    smooth_mask = cv2.GaussianBlur(mask_float, (blur_size, blur_size), 0)
    return smooth_mask

def process_image(img_path, save_path):
    image_bgr = cv2.imread(img_path)
    if image_bgr is None: return
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = image_rgb.shape[:2]

    # SAM 提示点设定
    offset_x, offset_y = int(w * 0.08), int(h * 0.08)
    cx, cy = w // 2, h // 2
    input_point = np.array([[cx, cy],[cx + offset_x, cy],[cx - offset_x, cy], [cx, cy + offset_y],[cx, cy - offset_y]])
    input_label = np.array([1, 1, 1, 1, 1])

    predictor.set_image(image_rgb)
    masks, _, _ = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )

    # === 执行完整的流水线 ===
    
    # 1. 获取初步掩码并填补内部空洞
    base_mask = refine_mask(masks, w, h)
    
    # 2. 【核心新增】二次处理：剪裁掉连接脆弱的残肢和杂物
    pruned_mask = prune_mask(base_mask, w, h)
    
    # 3. 边缘羽化抗锯齿
    final_alpha_mask = smooth_edges(pruned_mask, w, h)
    
    # ==========================

    y_indices, x_indices = np.where(pruned_mask > 0)
    if len(y_indices) == 0: return

    # 应用掩码
    alpha_mask_3d = np.expand_dims(final_alpha_mask, axis=2)
    masked_img = (image_bgr * alpha_mask_3d).astype(np.uint8)
    
    # 提取区域并裁剪
    y_min, y_max = y_indices.min(), y_indices.max()
    x_min, x_max = x_indices.min(), x_indices.max()
    pad = 5
    y_min, y_max = max(0, y_min - pad), min(h, y_max + pad)
    x_min, x_max = max(0, x_min - pad), min(w, x_max + pad)
    
    succulent_crop = masked_img[y_min:y_max+1, x_min:x_max+1]

    # 缩放并保持比例
    crop_h, crop_w = succulent_crop.shape[:2]
    scale = TARGET_SIZE / max(crop_w, crop_h)
    new_w, new_h = int(crop_w * scale), int(crop_h * scale)
    
    resized = cv2.resize(succulent_crop, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    # 放置到黑色画布
    final_canvas = np.zeros((TARGET_SIZE, TARGET_SIZE, 3), dtype=np.uint8)
    x_off, y_off = (TARGET_SIZE - new_w) // 2, (TARGET_SIZE - new_h) // 2
    final_canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized

    cv2.imwrite(save_path, final_canvas)

def main():
    if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
    subdirs = sorted([d for d in os.listdir(INPUT_DIR) if os.path.isdir(os.path.join(INPUT_DIR, d))])
    
    for subdir in subdirs:
        in_path = os.path.join(INPUT_DIR, subdir)
        out_path = os.path.join(OUTPUT_DIR, subdir)
        if not os.path.exists(out_path): os.makedirs(out_path)
        
        print(f"正在处理文件夹: {subdir}")
        imgs =[f for f in os.listdir(in_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        for name in tqdm(imgs):
            try:
                process_image(os.path.join(in_path, name), os.path.join(out_path, name))
            except Exception as e:
                print(f"跳过图片 {name}, 错误: {e}")

if __name__ == "__main__":
    main()