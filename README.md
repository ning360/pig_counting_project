# pig_counting_project
Automated pig counting for livestock channels. Uses YOLO + DeepSORT to detect, track, and count pigs in real time. Supports direction-aware counting (forward adds, reverse subtracts). Trained on 800 images + 5 videos from real farm conditions (occlusions, variable lighting).
# 检测单张图片
python task1_image_detection.py --input data/images/pig.jpg --output output/detection

# 检测图片文件夹
python task1_image_detection.py --input data/images/ --output output/detection

#视频计数
python main.py --video data/input_videos/pig_video.mp4 --output data/output_results/result

#web应用
streamlit run app.py

#移动端应用
streamlit run mobile_app.py
