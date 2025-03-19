from ultralytics import YOLO
import cv2
import argparse
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def main(args):
    # 加载模型
    model = YOLO(args.model)
    print(f"加载模型: {args.model}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 处理输入
    if os.path.isfile(args.source) and (args.source.endswith('.jpg') or args.source.endswith('.jpeg') or args.source.endswith('.png')):
        # 单张图片
        process_image(model, args.source, args)
    elif os.path.isfile(args.source) and (args.source.endswith('.mp4') or args.source.endswith('.avi')):
        # 视频
        process_video(model, args.source, args)
    elif os.path.isdir(args.source):
        # 目录中的所有图片
        for filename in os.listdir(args.source):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(args.source, filename)
                process_image(model, img_path, args)
    elif args.source == '0' or args.source.startswith('rtsp://') or args.source.startswith('http://'):
        # 摄像头或流
        process_stream(model, args.source, args)
    else:
        print(f"不支持的输入源: {args.source}")

def process_image(model, img_path, args):
    # 进行预测
    start_time = time.time()
    results = model.predict(
        source=img_path,
        conf=args.conf_threshold,
        iou=args.iou_threshold,
        imgsz=args.imgsz,
        device=args.device,
        save=False  # 不自动保存
    )
    inference_time = time.time() - start_time
    
    # 处理结果
    for r in results:
        boxes = r.boxes
        im_array = r.plot()  # 获取绘制结果
        
        # 添加额外信息
        img = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        text = f"推理时间: {inference_time*1000:.1f}ms | 检测到: {len(boxes)} 个坑洼"
        cv2.putText(img, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # 保存结果
        filename = os.path.basename(img_path)
        output_path = os.path.join(args.output_dir, f"pred_{filename}")
        plt.figure(figsize=(12, 8))
        plt.imshow(img)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        print(f"已保存预测结果到: {output_path}")
        print(f"检测到 {len(boxes)} 个坑洼，推理时间: {inference_time*1000:.1f}ms")

def process_video(model, video_path, args):
    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return
    
    # 获取视频信息
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 创建视频写入器
    filename = os.path.basename(video_path)
    output_path = os.path.join(args.output_dir, f"pred_{filename}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    total_time = 0
    
    print(f"开始处理视频: {video_path}")
    print(f"总帧数: {total_frames}, FPS: {fps}")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 每10帧显示一次进度
        if frame_count % 10 == 0:
            print(f"处理帧: {frame_count}/{total_frames} ({frame_count/total_frames*100:.1f}%)")
        
        # 预测当前帧
        start_time = time.time()
        results = model.predict(
            source=frame,
            conf=args.conf_threshold,
            iou=args.iou_threshold,
            imgsz=args.imgsz,
            device=args.device
        )
        inference_time = time.time() - start_time
        total_time += inference_time
        
        # 处理结果
        for r in results:
            boxes = r.boxes
            im_array = r.plot()  # 获取绘制结果
            
            # 添加信息
            text = f"推理时间: {inference_time*1000:.1f}ms | FPS: {1/inference_time:.1f} | 坑洼: {len(boxes)}"
            cv2.putText(im_array, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # 写入视频
            out.write(im_array)
        
        frame_count += 1
    
    # 释放资源
    cap.release()
    out.release()
    
    avg_time = total_time / frame_count if frame_count > 0 else 0
    avg_fps = 1 / avg_time if avg_time > 0 else 0
    
    print(f"视频处理完成: {output_path}")
    print(f"平均推理时间: {avg_time*1000:.1f}ms, 平均FPS: {avg_fps:.1f}")

def process_stream(model, source, args):
    # 打开视频流
    try:
        cap = cv2.VideoCapture(int(source) if source == '0' else source)
    except:
        print(f"无法打开视频流: {source}")
        return
    
    if not cap.isOpened():
        print(f"无法打开视频流: {source}")
        return
    
    # 获取视频信息
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"开始处理视频流. 按 'q' 键退出.")
    
    frame_count = 0
    start_time = time.time()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 预测
        results = model.predict(
            source=frame,
            conf=args.conf_threshold,
            iou=args.iou_threshold,
            imgsz=args.imgsz,
            device=args.device
        )
        
        # 处理结果
        for r in results:
            boxes = r.boxes
            im_array = r.plot()  # 获取绘制结果
            
            # 计算FPS
            frame_count += 1
            elapsed = time.time() - start_time
            fps = frame_count / elapsed
            
            # 添加信息
            text = f"FPS: {fps:.1f} | 检测到: {len(boxes)} 个坑洼"
            cv2.putText(im_array, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # 显示图像
            cv2.imshow("YOLOv8 推理", im_array)
        
        # 按q退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='YOLOv8 推理脚本')
    parser.add_argument('--model', type=str, required=True, help='模型权重路径')
    parser.add_argument('--source', type=str, default='0', help='输入源 (图片路径、视频路径、文件夹路径或摄像头ID)')
    parser.add_argument('--imgsz', type=int, default=640, help='输入图像尺寸')
    parser.add_argument('--conf-threshold', type=float, default=0.25, help='置信度阈值')
    parser.add_argument('--iou-threshold', type=float, default=0.45, help='NMS IoU阈值')
    parser.add_argument('--device', type=str, default='', help='推理设备 (例如 "0" 或 "cpu")')
    parser.add_argument('--output-dir', type=str, default='predictions', help='输出目录')
    
    args = parser.parse_args()
    main(args)