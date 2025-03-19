from ultralytics import YOLO
import argparse
import os
import cv2
import json
import random
import matplotlib.pyplot as plt
import numpy as np

def main(args):
    # 加载模型
    model = YOLO(args.model)
    print(f"加载模型: {args.model}")
    
    # 模型验证
    if args.data:
        print("开始验证...")
        results = model.val(
            data=args.data,
            imgsz=args.imgsz,
            batch=args.batch_size,
            device=args.device,
            split=args.split
        )
        
        # 打印验证结果
        metrics = results.box
        print(f"\n验证结果:")
        print(f"mAP@0.5: {metrics.map50:.4f}")
        print(f"mAP@0.5:0.95: {metrics.map:.4f}")
        print(f"精确率: {metrics.p:.4f}")
        print(f"召回率: {metrics.r:.4f}")
        print(f"F1分数: {metrics.f1:.4f}")
    
    # 可视化一些预测结果
    if args.test_images:
        visualize_predictions(model, args)

def visualize_predictions(model, args):
    # 数据路径
    data_path = "/data/baizy25/VOCdevkit/VOC2007"
    
    # 处理测试图片
    if args.test_images == "random":
        # 读取测试集JSON
        with open(os.path.join(data_path, "test.json"), "r") as f:
            test_data = json.load(f)
        
        # 随机选择图片
        test_images = random.sample(test_data["images"], min(5, len(test_data["images"])))
        img_paths = [os.path.join(data_path, "JPEGImages", img["file_name"]) for img in test_images]
    else:
        # 使用用户提供的图片路径
        img_paths = args.test_images.split(",")
    
    # 创建输出目录
    os.makedirs("predictions", exist_ok=True)
    
    # 对每张图片进行预测
    for i, img_path in enumerate(img_paths):
        if os.path.exists(img_path):
            # 进行预测
            results = model.predict(
                source=img_path,
                imgsz=args.imgsz,
                conf=args.conf_threshold,
                iou=args.iou_threshold,
                device=args.device
            )
            
            # 可视化结果
            for r in results:
                im_array = r.plot()  # 获取已绘制边界框的图像数组
                im = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)
                
                # 保存图片
                output_path = f"predictions/prediction_{i}.jpg"
                plt.figure(figsize=(10, 10))
                plt.imshow(im)
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(output_path)
                plt.close()
                print(f"保存预测结果到: {output_path}")
        else:
            print(f"找不到图片: {img_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='YOLOv8 验证脚本')
    parser.add_argument('--model', type=str, required=True, help='模型权重路径')
    parser.add_argument('--data', type=str, default='pothole_data.yaml', help='数据配置文件路径')
    parser.add_argument('--imgsz', type=int, default=640, help='输入图像尺寸')
    parser.add_argument('--batch-size', type=int, default=16, help='每批次样本数')
    parser.add_argument('--device', type=str, default='', help='推理设备 (例如 "0" 或 "cpu")')
    parser.add_argument('--conf-threshold', type=float, default=0.25, help='置信度阈值')
    parser.add_argument('--iou-threshold', type=float, default=0.45, help='NMS IoU阈值')
    parser.add_argument('--split', type=str, default='val', help='验证数据集分割 (test 或 val)')
    parser.add_argument('--test-images', type=str, default='random', help='测试图片路径，逗号分隔，或者"random"随机选择测试集图片')
    
    args = parser.parse_args()
    main(args)