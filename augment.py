import os
import cv2
import numpy as np
import random
from pathlib import Path

# ================= 配置参数 =================
INPUT_DIR = './dataset'
OUTPUT_DIR = './augmented_dataset'
MULTIPLIER = 2  # 增强倍数：1张原图 + 1张增强图 = 2张

# 增强参数
ANGLE_RANGE = (-180, 180)  # 随机旋转角度范围
SCALE_RANGE = (0.8, 1.1)   # 随机缩放比例 (0.8表示缩小并留黑边，1.1表示微微放大裁剪)
# ============================================

def get_random_augmentation(image):
    """对单张图片进行随机几何变换（旋转、缩放、翻转）"""
    h, w = image.shape[:2]
    center = (w / 2, h / 2)
    
    # 1. 生成随机旋转角度和缩放比例
    angle = random.uniform(*ANGLE_RANGE)
    scale = random.uniform(*SCALE_RANGE)
    
    # 2. 获取仿射变换矩阵
    M = cv2.getRotationMatrix2D(center, angle, scale)
    
    # 3. 应用仿射变换，空余部分用纯黑 (0,0,0) 填充，完美契合 SAM 的抠图
    aug_image = cv2.warpAffine(image, M, (w, h), borderValue=(0, 0, 0))
    
    # 4. 随机翻转 (增加镜像对称的多样性)
    flip_prob = random.random()
    if flip_prob < 0.25:
        aug_image = cv2.flip(aug_image, 1)  # 水平翻转
    elif flip_prob < 0.5:
        aug_image = cv2.flip(aug_image, 0)  # 垂直翻转
    elif flip_prob < 0.75:
        aug_image = cv2.flip(aug_image, -1) # 水平+垂直翻转
    # 剩下的 25% 概率不进行翻转
    
    return aug_image

def main():
    if not os.path.exists(INPUT_DIR):
        print(f"❌ 找不到输入目录: {INPUT_DIR}")
        return
        
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    total_processed = 0
    total_generated = 0
    
    # 遍历 class_000 到 class_014 的所有子文件夹
    for class_folder in sorted(os.listdir(INPUT_DIR)):
        class_path = os.path.join(INPUT_DIR, class_folder)
        
        # 忽略 README.md 等非文件夹文件
        if not os.path.isdir(class_path):
            continue
            
        # 在输出目录创建对应的类别子文件夹
        out_class_path = os.path.join(OUTPUT_DIR, class_folder)
        os.makedirs(out_class_path, exist_ok=True)
        
        images =[f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        print(f"📂 正在处理 {class_folder} ... (共 {len(images)} 张原图)")
        
        for img_name in images:
            img_path = os.path.join(class_path, img_name)
            image = cv2.imread(img_path)
            
            if image is None:
                print(f"⚠️ 无法读取图片: {img_path}")
                continue
                
            # 1. 保存原图 (或者叫基础图)
            base_name = os.path.splitext(img_name)[0]
            ext = os.path.splitext(img_name)[1]
            
            orig_out_path = os.path.join(out_class_path, f"{base_name}_orig{ext}")
            cv2.imwrite(orig_out_path, image)
            total_generated += 1
            
            # 2. 生成并保存增强后的图片
            for i in range(1, MULTIPLIER):
                aug_img = get_random_augmentation(image)
                aug_out_path = os.path.join(out_class_path, f"{base_name}_aug_{i}{ext}")
                cv2.imwrite(aug_out_path, aug_img)
                total_generated += 1
                
            total_processed += 1

    print("-" * 40)
    print(f"✅ 数据增强完成！")
    print(f"处理原图数量: {total_processed} 张")
    print(f"最终生成总图数: {total_generated} 张 (扩充了 {MULTIPLIER} 倍)")
    print(f"增强后的数据集已保存在: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()