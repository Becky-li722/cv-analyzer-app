import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import os
import numpy as np
from matplotlib import colormaps

# ========== 数据读取 ==========
def read_cv_data(csv_file):
    try:
        df = pd.read_csv(csv_file, encoding='latin1')
    except pd.errors.EmptyDataError:
        st.warning(f"⚠️ 文件 `{csv_file.name}` 是空的或无效，跳过处理。")
        return None, None, "V", "A"

    potential = df.get("x", None)
    current = df.get("y", None)

    x_unit = "V"
    y_unit = "A"

    if "x_unit" in df.columns:
        first_xu = df["x_unit"].dropna()
        if not first_xu.empty:
            x_unit = str(first_xu.iloc[0])
    if "y_unit" in df.columns:
        first_yu = df["y_unit"].dropna()
        if not first_yu.empty:
            y_unit = str(first_yu.iloc[0])

    return potential, current, x_unit, y_unit

def split_cycles_by_return_to_start(potential, tol=1e-3):

    potential = np.array(potential)
    v0 = potential[0]
    segments = []
    start_idx = 0

    # 从第2个点开始找，找那些电压值与v0相近的点
    for i in range(1, len(potential)):
        if abs(potential[i] - v0) < tol and (i - start_idx) > 10:
            # 发现“回到起点”且距离够长，切分一个圈
            segments.append((start_idx, i))
            start_idx = i
    # 最后一段剩余的数据
    if start_idx < len(potential):
        segments.append((start_idx, len(potential)))

    return segments

def select_cycles(segments, label="选择要显示的圈数"):
    cycle_labels = [f"Cycle {i+1}" for i in range(len(segments))]
    selected = st.multiselect(label, cycle_labels, default=cycle_labels)
    selected_idxs = [i for i, c in enumerate(cycle_labels) if c in selected]
    return selected_idxs

# ========== 单图绘制 ==========
def plot_single_cv(csv_file, label=None, smooth=False, mark_peaks=False, auto_split=False):
    potential, current, x_unit, y_unit = read_cv_data(csv_file)
    if potential is None or current is None:
        raise ValueError("CSV missing required columns")

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = colormaps.get_cmap('tab10')

    if auto_split:
        segments = split_cycles_by_return_to_start(potential)
        selected_cycles = select_cycles(segments, label="选择拆分出来的圈数")
        for i in selected_cycles:
            start, end = segments[i]
            color = colors(i % colors.N)
            ax.plot(potential[start:end], current[start:end], label=f"{label or os.path.basename(csv_file.name)} cycle {i+1}", color=color, linewidth=1)
            if smooth:
                x_smooth = savgol_filter(potential[start:end], 11, 3)
                y_smooth = savgol_filter(current[start:end], 11, 3)
                ax.plot(x_smooth, y_smooth, label=f"{label or os.path.basename(csv_file.name)} cycle {i+1} smoothed", color=color, linewidth=1)
    else:
        ax.plot(potential, current, label=f"{label or os.path.basename(csv_file.name)} (raw)", color='blue', alpha=0.6, linewidth=1)
        if smooth:
            x_smooth = savgol_filter(potential, 11, 3)
            y_smooth = savgol_filter(current, 11, 3)
            ax.plot(x_smooth, y_smooth, label=f"{label or os.path.basename(csv_file.name)} (smoothed)", color='red', linewidth=1)

    if mark_peaks and not auto_split:
        ox_idx = np.argmax(current)
        red_idx = np.argmin(current)
        ax.plot(potential[ox_idx], current[ox_idx], 'ro', label="Ox peak")
        ax.plot(potential[red_idx], current[red_idx], 'bo', label="Red peak")

    ax.set_xlabel(f"Potential ({x_unit})")
    ax.set_ylabel(f"Current ({y_unit})")
    ax.legend()
    ax.grid(True)
    return fig

# ========== 多图绘制 ==========
def plot_multi_cv(files, smooth=False, selected_files=None, auto_split=False):
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = colormaps.get_cmap('tab10')

    for i, f in enumerate(files):
        if selected_files and f.name not in selected_files:
            continue

        potential, current, x_unit, y_unit = read_cv_data(f)
        if potential is None or current is None:
            continue

        if auto_split:
            segments = split_cycles_by_return_to_start(potential)
            selected_cycles = select_cycles(segments, label=f"选择文件 {f.name} 的圈数")
            for j in selected_cycles:
                start, end = segments[j]
                color = colors((i + j) % colors.N)
                ax.plot(potential[start:end], current[start:end], label=f"{f.name} cycle {j+1}", color=color, linewidth=1)
                if smooth:
                    x_smooth = savgol_filter(potential[start:end], 11, 3)
                    y_smooth = savgol_filter(current[start:end], 11, 3)
                    ax.plot(x_smooth, y_smooth, label=f"{f.name} cycle {j+1} smoothed", color=color, linewidth=1)
        else:
            color = colors(i % colors.N)
            ax.plot(potential, current, label=f"{f.name.replace('.csv','')} (raw)", color=color, alpha=0.6, linewidth=1)
            if smooth:
                x_smooth = savgol_filter(potential, 11, 3)
                y_smooth = savgol_filter(current, 11, 3)
                ax.plot(x_smooth, y_smooth, label=f"{f.name.replace('.csv','')} (smoothed)", color=color, linewidth=1)

    if files:
        first_p, first_c, x_unit, y_unit = read_cv_data(files[0])
        if first_p is None or first_c is None:
            x_unit, y_unit = "V", "A"
    else:
        x_unit, y_unit = "V", "A"

    ax.set_xlabel(f"Potential ({x_unit})")
    ax.set_ylabel(f"Current ({y_unit})")
    ax.legend()
    ax.grid(True)
    return fig

st.set_page_config(page_title="CV曲线分析", layout="wide")
st.title("📉 CV 曲线可视化工具")

uploaded_files = st.file_uploader("上传一个或多个包含 x/y 列的 CSV 文件", type="csv", accept_multiple_files=True)

if uploaded_files:
    view_mode = st.radio("显示模式", ["📄 单文件查看", "📊 多图叠加"], horizontal=True)
    smooth = st.checkbox("平滑曲线（Savitzky-Golay）")
    mark_peaks = st.checkbox("标记峰值", disabled=(view_mode != "📄 单文件查看"))
    auto_split = st.checkbox("自动拆圈（分Cycle画多条曲线）")

    if view_mode == "📄 单文件查看":
        selected = st.selectbox("选择要显示的文件", uploaded_files, format_func=lambda f: f.name)
        fig = plot_single_cv(selected, smooth=smooth, mark_peaks=mark_peaks, auto_split=auto_split)
        st.pyplot(fig)

    else:
        st.markdown("### 选择要叠加的文件")
        selected_files = []
        for f in uploaded_files:
            if st.checkbox(f.name, value=True):
                selected_files.append(f.name)

        if selected_files:
            fig = plot_multi_cv(uploaded_files, smooth=smooth, selected_files=selected_files, auto_split=auto_split)
            st.pyplot(fig)
        else:
            st.info("请至少选择一个文件绘制图形")

else:
    st.info("请上传一个或多个 CSV 文件，文件应包含列名 `x`, `y`，以及可选的 `x_unit`, `y_unit`。")
st.markdown("<footer style='text-align:center; color:gray; font-size:12px; margin-top:50px;'>© 2025 Beckyli. All rights reserved.</footer>", unsafe_allow_html=True)