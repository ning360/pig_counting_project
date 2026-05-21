from ultralytics import YOLO
import torch
import gc

def memory_safe_training():
    """内存安全的训练配置"""
    
    print("🛡️ 使用内存安全配置开始训练...")
    
    # 强制垃圾回收
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # 检查内存状态
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        cached = torch.cuda.memory_reserved(0) / 1024**3
        print(f"🎮 GPU内存: {gpu_memory:.1f}GB, 已分配: {allocated:.1f}GB, 缓存: {cached:.1f}GB")
        device = 0
    else:
        print("💻 使用CPU训练")
        device = 'cpu'
    
    try:
        # 加载模型
        model = YOLO('yolov8n.pt')
        
        # 内存优化的训练配置
        results = model.train(
            data=r"C:\Users\34823\Desktop\pig_counting_project\yolotrain\dataset\pig_dataset.yaml",
            epochs=100,
            imgsz=320,          # 大幅减小图像尺寸
            batch=2,            # 很小的批量大小
            device=device,
            workers=1,          # 最少的工作进程
            lr0=0.01,
            patience=30,
            save=True,
            exist_ok=True,
            project='pig_memory_safe',
            name='pig_detector_safe',
            # 关闭内存密集型的数据增强
            mosaic=0.0,         # 关闭马赛克增强
            mixup=0.0,          # 关闭MixUp增强
            copy_paste=0.0,     # 关闭复制粘贴增强
            # 轻量级数据增强
            hsv_h=0.01,
            hsv_s=0.5,
            hsv_v=0.3,
            translate=0.05,
            scale=0.2,
            fliplr=0.3,
            # 其他优化
            close_mosaic=10,    # 提前关闭马赛克
            overlap_mask=False,
            mask_ratio=1,
            # 缓存设置
            cache=False,        # 不缓存数据集
            single_cls=True,    # 单类别训练
            verbose=True
        )
        
        print("✅ 训练完成!")
        return True
        
    except Exception as e:
        print(f"❌ 训练失败: {e}")
        return False

if __name__ == "__main__":
    memory_safe_training()
    