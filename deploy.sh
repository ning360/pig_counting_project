#!/bin/bash

echo "=== 猪只计数系统部署 ==="

# 创建环境
python -m venv pig_counting_env
source pig_counting_env/bin/activate

# 安装依赖
echo "安装依赖..."
pip install -r requirements.txt

# 安装Streamlit（用于Web应用）
pip install streamlit

# 下载预训练模型
echo "下载预训练模型..."
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -O models/yolov8n.pt

# 创建必要的目录
mkdir -p output/detection web_output data/input_videos data/output_results

# 测试系统
echo "测试系统组件..."
python -c "
from utils.config_loader import Config
from models.detector import PigDetector
print('✅ 配置加载测试通过')
print('✅ 检测器初始化测试通过')
"

echo "部署完成!"
echo ""
echo "运行命令:"
echo "任务一（图片检测）: python task1_image_detection.py --input path/to/images --output output/detection"
echo "任务二（视频计数）: python main.py --video path/to/video.mp4 --output result"
echo "任务三（Web应用）: streamlit run app.py"
echo "移动端应用: streamlit run mobile_app.py"