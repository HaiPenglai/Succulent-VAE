# --- START OF FILE fusion.py ---
import os
import torch
import random
import argparse
import numpy as np
from PIL import Image
from torchvision import transforms
from diffusers import AutoencoderKL

def main():
    # --- 1. 命令行参数配置 ---
    parser = argparse.ArgumentParser(description="多肉植物 VAE 特征融合测试")
    parser.add_argument("--epoch", type=int, default=200, help="选择要读取的模型 Epoch 轮数 (默认: 200)")
    parser.add_argument("--num_pairs", type=int, default=5, help="要生成的随机融合组数 (默认: 5)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # --- 2. 动态定位模型路径 ---
    model_dir = "succulent_vae_128_model"
    # 兼容两种保存格式: checkpoint-epoch-200 或 checkpoint-epoch-0200
    checkpoint_path = os.path.join(model_dir, f"checkpoint-epoch-{args.epoch:03d}")
    if not os.path.exists(checkpoint_path):
        checkpoint_path = os.path.join(model_dir, f"checkpoint-epoch-{args.epoch}")
        
    if not os.path.exists(checkpoint_path):
        print(f"❌ 找不到模型权重: {checkpoint_path}")
        print("💡 请确认该 epoch 是否已保存，或者尝试使用 'latest' 文件夹作为路径。")
        return

    print(f"🚀 正在加载第 {args.epoch} 轮的模型: {checkpoint_path}")
    vae = AutoencoderKL.from_pretrained(checkpoint_path).to(device)
    vae.eval()

    # --- 3. 加载原始数据集 ---
    # 使用原 dataset 而不是 augmented_dataset，保证原图视角是正的，融合效果更好看
    dataset_path = "dataset"
    all_images =[]
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                all_images.append(os.path.join(root, file))
    
    if len(all_images) < 2:
        print("❌ 数据集图片不足，无法进行融合！")
        return

    preprocess = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    # --- 4. 核心融合逻辑 ---
    output_folder = "fusion_results"
    os.makedirs(output_folder, exist_ok=True)
    print(f"✨ 开始进行特征融合，共生成 {args.num_pairs} 组...")

    # 保存所有行，最后拼接成一张展示大图
    final_grid_rows =[]

    for i in range(args.num_pairs):
        # 随机挑选两张多肉图片 (A 和 B)
        img_a_path, img_b_path = random.sample(all_images, 2)
        
        img_a = Image.open(img_a_path).convert("RGB")
        img_b = Image.open(img_b_path).convert("RGB")
        
        tensor_a = preprocess(img_a).unsqueeze(0).to(device)
        tensor_b = preprocess(img_b).unsqueeze(0).to(device)

        with torch.no_grad():
            # 编码到潜空间 (获取核心特征 8x8 latent)
            latent_a = vae.encode(tensor_a).latent_dist.mode()
            latent_b = vae.encode(tensor_b).latent_dist.mode()
            
            # 🌟 核心：在潜空间中进行 1:1 的特征加权平均
            latent_c = 0.5 * latent_a + 0.5 * latent_b
            
            # 解码生成新的多肉 (植物C)
            tensor_c = vae.decode(latent_c).sample

        # 辅助函数：Tensor 还原为图片
        def tensor_to_pil(t):
            t = (t / 2 + 0.5).clamp(0, 1) # 反归一化到 [0, 1]
            # 🌟 修复 Bug：必须先乘以 255，再转换成 uint8！
            t_np = t.permute(0, 2, 3, 1).cpu().numpy()[0]
            t_np = (t_np * 255).round().astype(np.uint8)
            return Image.fromarray(t_np)

        pil_a = tensor_to_pil(tensor_a)
        pil_b = tensor_to_pil(tensor_b)
        pil_c = tensor_to_pil(tensor_c)

        # 🖼️ 拼图排版： [植物A]  ->  [融合多肉C]  <-[植物B]
        # 画布大小：宽 128*3，高 128
        row_image = Image.new('RGB', (128 * 3, 128))
        row_image.paste(pil_a, (0, 0))            # 左边：植物 A
        row_image.paste(pil_c, (128, 0))          # 中间：融合植物 C
        row_image.paste(pil_b, (128 * 2, 0))      # 右边：植物 B
        
        # 保存单组结果
        out_name = os.path.join(output_folder, f"fusion_pair_{i+1}.png")
        row_image.save(out_name)
        final_grid_rows.append(np.array(row_image))
        
    # 拼成一张汇总大图，发给用户最震撼！
    final_grid = np.vstack(final_grid_rows)
    Image.fromarray(final_grid).save(os.path.join(output_folder, "00_all_fusions_summary.png"))
    
    print("-" * 40)
    print("✅ 融合完美完成！")
    print(f"📁 结果已保存在目录: ./{output_folder}/")
    print("💡 '00_all_fusions_summary.png' 可以概览结果！")

if __name__ == "__main__":
    main()
# --- END OF FILE fusion.py ---