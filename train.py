from ultralytics import YOLO
import os
import torch
import argparse

def main(args):
    # 打印系统信息
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU设备: {torch.cuda.get_device_name(0)}")
    
    # 创建结果目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化模型
    if os.path.exists(args.model):
        # 如果是已有模型路径，加载模型
        model = YOLO(args.model)
        print(f"加载预训练模型: {args.model}")
    else:
        # 否则使用预训练的YOLOv8模型
        model = YOLO(f"{args.model}.pt")
        print(f"使用官方预训练模型: {args.model}")
    
    # 训练模型
    print("开始训练...")
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch_size,
        device=args.device,
        workers=args.workers,
        project=args.output_dir,
        name=args.exp_name,
        exist_ok=True,
        pretrained=True,
        optimizer=args.optimizer,
        patience=args.patience,
        lr0=args.learning_rate,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='YOLOv8 训练脚本')
    parser.add_argument('--model', type=str, default='yolov11m', help='模型类型 (yolov8n, yolov8s, yolov8m, yolov8l, yolov8x) 或模型路径')
    parser.add_argument('--data', type=str, default='pothole_data.yaml', help='数据配置文件路径')
    parser.add_argument('--epochs', type=int, default=300, help='训练轮次')
    parser.add_argument('--imgsz', type=int, default=640, help='输入图像尺寸')
    parser.add_argument('--batch-size', type=int, default=32, help='每批次样本数')
    parser.add_argument('--device', type=str, default='0,1', help='训练设备 (例如 "0" 或 "cpu")')
    parser.add_argument('--workers', type=int, default=8, help='数据加载器worker数量')
    parser.add_argument('--output-dir', type=str, default='runs', help='输出目录')
    parser.add_argument('--exp-name', type=str, default='pothole_detection_v11', help='实验名称')
    parser.add_argument('--optimizer', type=str, default='auto', help='优化器 (auto, SGD, Adam, AdamW, RMSProp)')
    parser.add_argument('--patience', type=int, default=50, help='早停patience')
    parser.add_argument('--learning-rate', type=float, default=0.01, help='初始学习率')
    
    args = parser.parse_args()
    main(args)