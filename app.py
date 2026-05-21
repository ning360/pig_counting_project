#!/usr/bin/env python3
"""
任务三：猪只计数Web应用
使用Streamlit开发，支持视频上传和实时处理
"""

import streamlit as st
import tempfile
import os
import cv2
from main import PigCountingSystem
import time

def main():
    st.set_page_config(
        page_title="猪只计数系统",
        page_icon="🐷",
        layout="wide"
    )
    
    st.title("🐷 猪只计数系统")
    st.markdown("上传通道猪只视频，系统将自动检测和计数")
    
    # 文件上传
    uploaded_file = st.file_uploader(
        "选择视频文件", 
        type=['mp4', 'avi', 'mov', 'flv'],
        help="支持MP4、AVI、MOV、FLV格式"
    )
    
    if uploaded_file is not None:
        # 保存上传的文件
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            video_path = tmp_file.name
        
        # 配置选项
        col1, col2 = st.columns(2)
        
        with col1:
            config_option = st.selectbox(
                "选择处理配置",
                ["default.yaml", "fast_inference.yaml", "gpu_optimized.yaml"],
                help="选择适合您硬件的配置"
            )
        
        with col2:
            show_processing = st.checkbox("显示处理过程", value=True)
        
        # 处理按钮
        if st.button("开始处理视频", type="primary"):
            with st.spinner("正在处理视频，请稍候..."):
                try:
                    # 初始化系统
                    system = PigCountingSystem(f"configs/{config_option}")
                    
                    # 创建输出文件
                    output_dir = "web_output"
                    os.makedirs(output_dir, exist_ok=True)
                    output_path = os.path.join(output_dir, "processed_video")
                    
                    # 处理视频
                    start_time = time.time()
                    results = system.process_video(video_path, output_path)
                    processing_time = time.time() - start_time
                    
                    # 显示结果
                    st.success("处理完成!")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("总计数", results['final_count'])
                    
                    with col2:
                        st.metric("处理帧数", results['total_frames'])
                    
                    with col3:
                        st.metric("处理时间", f"{processing_time:.2f}秒")
                    
                    # 显示处理后的视频
                    if show_processing:
                        st.subheader("处理结果视频")
                        processed_video_path = f"{output_path}.mp4"
                        
                        if os.path.exists(processed_video_path):
                            with open(processed_video_path, 'rb') as video_file:
                                video_bytes = video_file.read()
                            
                            st.video(video_bytes)
                        else:
                            st.warning("处理后的视频文件未找到")
                    
                    # 显示统计信息
                    st.subheader("统计信息")
                    if results.get('results'):
                        frame_counts = [r['total_count'] for r in results['results']]
                        frame_indices = [r['frame_idx'] for r in results['results']]
                        
                        # 绘制计数趋势图
                        st.line_chart(
                            data={ '猪只数量': frame_counts },
                            use_container_width=True
                        )
                    
                except Exception as e:
                    st.error(f"处理过程中出现错误: {str(e)}")
                
                finally:
                    # 清理临时文件
                    if os.path.exists(video_path):
                        os.unlink(video_path)
    
    # 使用说明
    with st.expander("使用说明"):
        st.markdown("""
        ### 系统功能
        1. **视频上传**: 支持常见视频格式
        2. **智能计数**: 自动检测和跟踪猪只
        3. **方向判断**: 区分正常方向和反向通过
        4. **轨迹展示**: 显示猪只运动轨迹
        
        ### 计数规则
        - ✅ 正常方向（左下→右上）：计数 +1
        - 🔄 反方向（右上→左下）：计数 -1
        - 📊 最终显示净计数结果
        """)

if __name__ == "__main__":
    main()