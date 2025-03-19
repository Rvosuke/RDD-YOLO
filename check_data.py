import os
import random
import matplotlib.pyplot as plt
import cv2
import numpy as np
from pathlib import Path

def visualize_yolo_dataset(dataset_dir, num_samples=3):
    """可视化YOLO数据集中的一些样本"""
    dataset_dir = Path(dataset_dir)
    images_dir = dataset_dir / "images" / "train"
    labels_dir = dataset_dir / "labels" / "train"
    
    # 获取所有有标签的图像
    image_files = []
    for img_file in os.listdir(images_dir):
        label_file = labels_dir / f"{Path(img_file).stem}.txt"
        if label_file.exists():
            image_files.append(img_file)
    
    if not image_files:
        print("没有找到带标签的图像!")
        return
    
    # 随机选择样本
    samples = random.sample(image_files, min(num_samples, len(image_files)))
    
    # 创建图表
    fig, axes = plt.subplots(1, len(samples), figsize=(15, 5))
    if len(samples) == 1:
        axes = [axes]
    
    # 可视化每个样本
    for i, img_file in enumerate(samples):
        # 读取图像
        img_path = images_dir / img_file
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"无法读取图像: {img_path}")
            continue
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width = img.shape[:2]
        
        # 读取标签
        label_file = labels_dir / f"{Path(img_file).stem}.txt"
        boxes = []
        if label_file.exists():
            with open(label_file, 'r') as f:
                for line in f:
                    data = line.strip().split()
                    if len(data) == 5:
                        class_id = int(data[0])
                        x_center = float(data[1]) * width
                        y_center = float(data[2]) * height
                        box_width = float(data[3]) * width
                        box_height = float(data[4]) * height
                        
                        # 计算左上角坐标
                        x1 = int(x_center - box_width / 2)
                        y1 = int(y_center - box_height / 2)
                        # 计算右下角坐标
                        x2 = int(x_center + box_width / 2)
                        y2 = int(y_center + box_height / 2)
                        
                        boxes.append((x1, y1, x2, y2, class_id))
        
        # 绘制边界框
        for (x1, y1, x2, y2, class_id) in boxes:
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(img, f"pothole", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # 显示图像
        axes[i].imshow(img)
        axes[i].set_title(f"{img_file}\n{len(boxes)} boxes")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig("yolo_dataset_preview.png")
    plt.close()
    print("预览图像已保存到 yolo_dataset_preview.png")

    # 统计信息
    total_images = len(os.listdir(images_dir))
    labeled_images = len(image_files)
    total_labels = sum(1 for img in image_files for _ in open(labels_dir / f"{Path(img).stem}.txt", 'r'))
    
    print(f"数据集统计:")
    print(f"图像总数: {total_images}")
    print(f"带标签的图像数: {labeled_images}")
    print(f"标签总数: {total_labels}")

    # 检查标签格式
    is_valid = True
    for img in image_files[:10]:  # 只检查前10张图像
        label_file = labels_dir / f"{Path(img).stem}.txt"
        with open(label_file, 'r') as f:
            for i, line in enumerate(f):
                parts = line.strip().split()
                if len(parts) != 5:
                    print(f"警告: {label_file} 第 {i+1} 行格式错误")
                    is_valid = False
                else:
                    try:
                        class_id = int(parts[0])
                        x, y, w, h = map(float, parts[1:])
                        if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                            print(f"警告: {label_file} 第 {i+1} 行坐标超出范围 [{x}, {y}, {w}, {h}]")
                            is_valid = False
                    except ValueError:
                        print(f"警告: {label_file} 第 {i+1} 行数值格式错误")
                        is_valid = False
    
    if is_valid:
        print("所有检查的标签格式正确")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="检查YOLO格式数据集")
    parser.add_argument("--dir", type=str, default="~/projects/yolo/dataset", help="数据集目录")
    parser.add_argument("--samples", type=int, default=3, help="要可视化的样本数量")
    
    args = parser.parse_args()
    dataset_dir = os.path.expanduser(args.dir)
    
    visualize_yolo_dataset(dataset_dir, args.samples)