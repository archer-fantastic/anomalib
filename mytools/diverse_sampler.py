import os
import shutil
import torch
import numpy as np
from PIL import Image
from torchvision import models
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
from tqdm import tqdm

def get_leaf_directories(root_dir):
    """获取所有最底层的子文件夹（包含图片文件但没有子文件夹）"""
    leaf_dirs = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if not dirnames:
            image_files = [f for f in filenames if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
            if image_files:
                leaf_dirs.append((dirpath, image_files))
    return leaf_dirs

def extract_features(image_paths, model, transform, device, batch_size=32):
    """使用ResNet50提取图像的池化层特征（Softmax前）"""
    features = []
    valid_paths = []
    model.eval()
    with torch.no_grad():
        # 添加提取特征的进度条
        for i in tqdm(range(0, len(image_paths), batch_size), desc="提取特征", leave=False):
            batch_paths = image_paths[i:i+batch_size]
            batch_tensors = []
            
            for path in batch_paths:
                try:
                    img = Image.open(path).convert('RGB')
                    batch_tensors.append(transform(img))
                    valid_paths.append(path)
                except Exception as e:
                    print(f"读取图片失败 {path}: {e}")
                    
            if not batch_tensors:
                continue
                
            batch_tensors = torch.stack(batch_tensors).to(device)
            # 提取特征，输出维度为 (N, 2048)
            batch_features = model(batch_tensors)
            features.append(batch_features.cpu().numpy())
            
    if features:
        return np.vstack(features), valid_paths
    return np.array([]), []

def diverse_sampling(features, num_samples):
    """基于K-Means聚类进行多样性采样，返回选中样本的索引"""
    if num_samples >= len(features):
        return list(range(len(features)))
    if num_samples <= 0:
        return []
        
    # 需求：聚类数量不确定，但要求保留20%且最不相似。
    # 最佳方案：将K-Means的 K 值直接设为我们要采样的数量 (总数的20%)
    kmeans = KMeans(n_clusters=num_samples, random_state=42, n_init="auto")
    kmeans.fit(features)
    
    # 获取聚类中心
    centers = kmeans.cluster_centers_
    # 计算所有样本到所有聚类中心的距离
    distances = pairwise_distances(centers, features)
    
    selected_indices = []
    # 添加聚类采样的进度条
    for i in tqdm(range(num_samples), desc="聚类采样", leave=False):
        # 找到距离当前聚类中心最近的真实样本图像
        closest_idx = np.argmin(distances[i])
        # 如果该样本已被其他中心选中，则取次近的
        while closest_idx in selected_indices:
            distances[i, closest_idx] = np.inf
            closest_idx = np.argmin(distances[i])
        selected_indices.append(closest_idx)
        
    return selected_indices

def main(input_dir, output_dir, sample_ratio=0.2):
    # 1. 初始化设备和预训练模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    
    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    # 核心：将最后的全连接层替换为Identity，直接输出 Global Average Pooling 后的2048维向量
    model.fc = torch.nn.Identity()
    model = model.to(device)
    
    transform = weights.transforms()
    
    # 2. 扫描最底层的子文件夹
    leaf_dirs = get_leaf_directories(input_dir)
    print(f"共找到 {len(leaf_dirs)} 个最底层包含图片的文件夹")
    
    # 3. 逐个文件夹提取特征并采样
    for dirpath, filenames in tqdm(leaf_dirs, desc="处理文件夹"):
        image_paths = [os.path.join(dirpath, f) for f in filenames]
        num_images = len(image_paths)
        num_samples = max(1, int(num_images * sample_ratio))
        
        print(f"\n-> 正在处理: {dirpath} | 图片: {num_images}张 | 采样: {num_samples}张")
        
        features, valid_paths = extract_features(image_paths, model, transform, device)
        if len(features) == 0:
            continue
            
        selected_indices = diverse_sampling(features, num_samples)
        
        # 4. 将选中的图像拷贝到目标目录（保持原目录结构，相同前缀自动合并）
        # 使用 rel_path 提取相对于 INPUT_DIR 的子目录结构
        rel_path = os.path.relpath(dirpath, input_dir)
        # 将子目录结构拼接到 OUTPUT_DIR，实现前缀替换
        target_dir = os.path.join(output_dir, rel_path)
        
        # exist_ok=True 保证了如果目标文件夹已存在，不会报错而是直接将文件合并进去
        os.makedirs(target_dir, exist_ok=True)
        
        for idx in selected_indices:
            src_path = valid_paths[idx]
            dst_path = os.path.join(target_dir, os.path.basename(src_path))
            shutil.copy2(src_path, dst_path)
            
    print(f"\n✅ 采样完成！采样结果已保存至: {output_dir}")

if __name__ == "__main__":
    # 请在此处配置你的数据集输入输出路径
    INPUT_DIR = r"Z:\14-调试数据\lxm\Dataset\Anomalib\TB\OK_Full"
    OUTPUT_DIR = r"Z:\14-调试数据\lxm\Dataset\Anomalib\TB\OK_V2"
    
    main(INPUT_DIR, OUTPUT_DIR, sample_ratio=0.1)
