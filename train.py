# --- START OF FILE train.py ---
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from diffusers import AutoencoderKL
from diffusers.optimization import get_cosine_schedule_with_warmup
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm.auto import tqdm
import torchvision

# --- 配置类 ---
class Config:
    dataset_path = "augmented_dataset"
    output_dir = "succulent_vae_128_model"
    image_size = 128                # 🌟 128 最快
    
    # 5090 在 128 分辨率下可以开启疯狂模式
    train_batch_size = 128          # 🌟 大 batch size，跑得飞快
    learning_rate = 2e-4            # batch 大了，学习率稍微调高一点点
    num_epochs = 200
    save_every_epochs = 10          # 🌟 每 10 轮存一次，也就几分钟的事
    lr_warmup_steps = 500
    
    mixed_precision = "bf16"
    num_workers = 8

    # 保持 VGG 损失，128 下用了 VGG 会让多肉显得非常有质感
    kl_weight = 1e-6 
    perceptual_weight = 1.0
    mse_weight = 1.0

# ---  VGG 感知损失 (Perceptual Loss) ---
class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # 加载预训练的 VGG16 特征层
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        # 选取第 4, 9, 16, 23 层作为特征提取层 (涵盖浅层纹理和深层语义)
        self.blocks = nn.ModuleList([vgg[:4], vgg[4:9], vgg[9:16], vgg[16:23]])
        for p in self.parameters():
            p.requires_grad = False
            
        # ImageNet 标准化参数
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input_x, target_x):
        # 将 [-1, 1] 的图像转换回 [0, 1]
        input_x = (input_x + 1) / 2
        target_x = (target_x + 1) / 2
        
        # ImageNet 标准化
        input_x = (input_x - self.mean) / self.std
        target_x = (target_x - self.mean) / self.std
        
        loss = 0.0
        x, y = input_x, target_x
        for block in self.blocks:
            x = block(x)
            y = block(y)
            # 使用 L1 Loss 计算特征图之间的差异
            loss += F.l1_loss(x, y)
        return loss

# --- 数据集加载 (遍历所有子文件夹) ---
class SucculentDataset(Dataset):
    def __init__(self, root, transform):
        self.images =[]
        # 递归遍历所有子文件夹 (class_000 到 class_014)
        for path, subdirs, files in os.walk(root):
            for name in files:
                if name.lower().endswith(('.jpg', '.png', '.jpeg')):
                    self.images.append(os.path.join(path, name))
        self.transform = transform
        print(f"✅ 成功从 {root} 加载了 {len(self.images)} 张多肉图片。")

    def __len__(self): return len(self.images)

    def __getitem__(self, i):
        image = Image.open(self.images[i]).convert("RGB")
        return self.transform(image)

def train():
    config = Config()
    accelerator = Accelerator(mixed_precision=config.mixed_precision)

    # 🌟 1. 标准 AutoencoderKL (稳定扩散的同款架构)
    model = AutoencoderKL(
        in_channels=3,
        out_channels=3,
        down_block_types=("DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D", "DownEncoderBlock2D"),
        up_block_types=("UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D", "UpDecoderBlock2D"),
        block_out_channels=(128, 256, 512, 512), # 强大的通道数，支撑 256 分辨率的细节
        latent_channels=4,                       # 压缩到 4 通道的潜空间 (高度压缩，利于融合)
        layers_per_block=2,
    )
    
    # 开启梯度检查点节省显存
    model.enable_gradient_checkpointing()

    # 初始化损失函数和优化器
    perceptual_loss_fn = VGGPerceptualLoss().to(accelerator.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)

    # 数据预处理
    preprocess = transforms.Compose([
        transforms.Resize((config.image_size, config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]), # 归一化到 [-1, 1]
    ])
    
    dataset = SucculentDataset(config.dataset_path, preprocess)
    train_dataloader = DataLoader(dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)

    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer, num_warmup_steps=config.lr_warmup_steps, num_training_steps=(len(train_dataloader) * config.num_epochs),
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    # 创建保存目录
    if accelerator.is_main_process:
        os.makedirs("samples_vae", exist_ok=True)
        os.makedirs(config.output_dir, exist_ok=True)
        # 固定一批测试数据，用于直观对比每一轮的重建效果
        fixed_test_images = next(iter(train_dataloader))[:8].to(accelerator.device)

    print(f"🚀 开始 VAE 训练... 目标轮数: {config.num_epochs}")

    for epoch in range(config.num_epochs):
        model.train()
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch+1}/{config.num_epochs}")

        for step, clean_images in enumerate(train_dataloader):
            # 1. 前向传播：编码 (得到均值和方差)
            posterior = model.encode(clean_images).latent_dist
            # 2. 从潜空间采样 (重参数化技巧)
            z = posterior.sample()
            # 3. 解码：还原图像
            reconstructed = model.decode(z).sample

            # --- 🌟 计算联合损失 ---
            # 1. MSE 像素损失
            loss_mse = F.mse_loss(reconstructed, clean_images)
            
            # 2. VGG 感知损失 (衡量纹理、质感差异)
            loss_perceptual = perceptual_loss_fn(reconstructed, clean_images)
            
            # 3. KL 散度 (约束潜空间分布，让它符合标准正态分布，方便后期特征融合)
            loss_kl = posterior.kl().mean()

            # 总损失
            loss = (config.mse_weight * loss_mse) + \
                   (config.perceptual_weight * loss_perceptual) + \
                   (config.kl_weight * loss_kl)
            
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            progress_bar.update(1)
            progress_bar.set_postfix({
                "mse": loss_mse.item(),
                "perc": loss_perceptual.item(),
                "kl": loss_kl.item()
            })
        
        # --- 🌟 定期保存和测试 (重建能力 + 生成能力) ---
        if (epoch + 1) % config.save_every_epochs == 0 and accelerator.is_main_process:
            model.eval()
            print(f"\n✨ 正在生成 Epoch {epoch+1} 的测试图...")
            with torch.no_grad():
                # 【测试 1：重建测试】(对比原图和生成的图)
                posterior = model.encode(fixed_test_images).latent_dist
                recon_images = model.decode(posterior.sample()).sample
                
                # 将 原图 和 重建图 拼接在一起 (上排原图，下排重建)
                comparison = torch.cat([fixed_test_images, recon_images], dim=0)
                comparison = (comparison / 2 + 0.5).clamp(0, 1) # 恢复到 [0,1]
                grid = torchvision.utils.make_grid(comparison, nrow=8)
                torchvision.utils.save_image(grid, f"samples_vae/epoch_{epoch+1:03d}_reconstruction.png")

                # 【测试 2：无中生有/融合测试】(从标准正态分布的潜空间直接抽卡生成全新的多肉)
                # 潜空间的尺寸:[Batch, latent_channels, H/8, W/8]
                random_latent = torch.randn(8, 4, config.image_size // 8, config.image_size // 8).to(accelerator.device)
                generated_images = model.decode(random_latent).sample
                generated_images = (generated_images / 2 + 0.5).clamp(0, 1)
                grid_gen = torchvision.utils.make_grid(generated_images, nrow=4)
                torchvision.utils.save_image(grid_gen, f"samples_vae/epoch_{epoch+1:03d}_random_gen.png")

            # 保存模型权重 (使用 diffusers 原生格式)
            checkpoint_dir = os.path.join(config.output_dir, f"checkpoint-epoch-{epoch+1:03d}")
            accelerator.unwrap_model(model).save_pretrained(checkpoint_dir)
            # 保存 latest 版本
            accelerator.unwrap_model(model).save_pretrained(os.path.join(config.output_dir, "latest"))
            
            print(f"✅ 模型已更新至: {config.output_dir}/latest")
            print(f"✅ 测试图已保存至: samples_vae/ 文件夹")

    print("🎉  VAE 训练圆满完成！")

if __name__ == "__main__":
    train()
# --- END OF FILE train.py ---