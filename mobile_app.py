#!/usr/bin/env python3
"""
移动端适配的猪只计数应用
使用Streamlit开发，针对手机屏幕优化
"""

import streamlit as st
import tempfile
import os
from main import PigCountingSystem

def mobile_main():
    # 移动端配置
    st.set_page_config(
        page_title="猪只计数",
        page_icon="🐷",
        layout="centered",
        initial_sidebar_state="collapsed"
    )
    
    # 移动端样式
    st.markdown("""
    <style>
    .main > div {
        padding-top: 1rem;
    }
    .stButton button {
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)
    
    st.header("🐷 猪只计数")
    st.caption("智慧养猪 - 通道计数系统")
    
    # 简化版文件上传
    uploaded_file = st.file_uploader(
        "上传视频",
        type=['mp4', 'mov'],
        help="选择通道猪只视频"
    )
    
    if uploaded_file is not None:
        # 显示视频预览
        st.video(uploaded_file)
        
        if st.button("开始计数", type="primary"):
            with st.spinner("分析中..."):
                try:
                    # 保存临时文件
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                        tmp_file.write(uploaded_file.getvalue())
                        video_path = tmp_file.name
                    
                    # 快速处理
                    system = PigCountingSystem("configs/fast_inference.yaml")
                    results = system.process_video(video_path, None)
                    
                    # 显示结果
                    st.success("分析完成!")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("总计数", results['final_count'])
                    with col2:
                        st.metric("处理帧数", results['total_frames'])
                    
                except Exception as e:
                    st.error(f"处理失败: {str(e)}")
                finally:
                    if os.path.exists(video_path):
                        os.unlink(video_path)

if __name__ == "__main__":
    mobile_main()