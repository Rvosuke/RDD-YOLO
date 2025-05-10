from ultralytics import YOLO
import os
import torch
import argparse
import math

def test_model(model, data_yaml, imgsz, batch_size, device, epoch):
    """在特定epoch执行完整测试并打印结果"""
    print(f"\n--- 执行第 {epoch} 轮的完整测试 ---")
    results = model.val(data=data_yaml, 
                        imgsz=imgsz, 
                        batch=batch_size, 
                        device=device)
    print(f"测试结果 (Epoch {epoch}):")
    print(f"mAP@0.5: {results.box.map50:.4f}")
    print(f"mAP@0.5:0.95: {results.box.map:.4f}")
    return results

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
        model = YOLO(args.model)
        print(f"使用官方预训练模型: {args.model}")
    
    # 计算保存检查点的间隔
    save_period = math.ceil(args.epochs / args.num_checkpoints)
    print(f"将每 {save_period} 轮保存一次检查点，总共保存约 {args.num_checkpoints} 个")
    
    # 训练模型
    print("开始训练...")
    results = model.train(
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
        save_period=save_period,  # 定期保存检查点
    )
    
    # 如果需要在每个保存点执行测试
    if args.test_checkpoints:
        print("\n开始测试所有保存的检查点...")
        checkpoint_dir = os.path.join(args.output_dir, args.exp_name, 'weights')
        checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pt') and 'epoch' in f]
        checkpoints.sort(key=lambda x: int(x.split('_')[1].split('.')[0]))
        
        for checkpoint in checkpoints:
            checkpoint_path = os.path.join(checkpoint_dir, checkpoint)
            epoch = int(checkpoint.split('_')[1].split('.')[0])
            print(f"\n测试检查点: {checkpoint}")
            test_model = YOLO(checkpoint_path)
            test_model.val(data=args.data, imgsz=args.imgsz, batch=args.batch_size, device=args.device)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='YOLOv8 训练脚本')
    parser.add_argument('--model', type=str, default='yolo11n', help='模型类型或模型路径')
    parser.add_argument('--data', type=str, default='data.yaml', help='数据配置文件路径')
    parser.add_argument('--epochs', type=int, default=300, help='训练轮次')
    parser.add_argument('--imgsz', type=int, default=640, help='输入图像尺寸')
    parser.add_argument('--batch-size', type=int, default=128, help='每批次样本数')
    parser.add_argument('--device', type=str, default='0,1', help='训练设备 (例如 "0" 或 "cpu")')
    parser.add_argument('--workers', type=int, default=8, help='数据加载器worker数量')
    parser.add_argument('--output-dir', type=str, default='runs', help='输出目录')
    parser.add_argument('--exp-name', type=str, default='motor', help='实验名称')
    parser.add_argument('--optimizer', type=str, default='auto', help='优化器 (auto, SGD, Adam, AdamW, RMSProp)')
    parser.add_argument('--patience', type=int, default=50, help='早停patience')
    parser.add_argument('--learning-rate', type=float, default=0.01, help='初始学习率')
    parser.add_argument('--num-checkpoints', type=int, default=10, help='要保存的检查点数量')
    parser.add_argument('--test-checkpoints', action='store_true', help='是否测试所有保存的检查点')
    
    args = parser.parse_args()
    main(args)