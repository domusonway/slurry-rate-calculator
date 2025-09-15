import cv2
import numpy as np
import streamlit as st
# from sklearn.cluster import KMeans
from streamlit_image_comparison import image_comparison
from streamlit_drawable_canvas import st_canvas

st.set_page_config(layout="wide")
st.title("满浆率计算")

# ========== 左侧控制面板 ==========
st.sidebar.header("输入")
uploaded_file = st.sidebar.file_uploader("上传图像", type=["jpg", "png", "bmp"])

algo = st.sidebar.selectbox(
    "选择二值化算法",
    ["全局阈值", "Otsu", "自适应阈值", "K-means 聚类", "GrabCut 交互式"]
)

thresh_val = st.sidebar.slider("阈值/参数调节", 0, 255, 128)

# ========== 二值化函数 ==========
def binarize(img, algo, val, roi_mask=None):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if algo == "全局阈值":
        _, binary = cv2.threshold(gray, val, 255, cv2.THRESH_BINARY)

    elif algo == "Otsu":
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    elif algo == "自适应阈值":
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY, blockSize=val if val % 2 == 1 else val+1, C=2
        )

    elif algo == "K-means 聚类":
        print("聚类算法暂未实现")
        # Z = gray.reshape((-1,1)).astype(np.float32)
        # kmeans = KMeans(n_clusters=2, n_init=10).fit(Z)
        # labels = kmeans.labels_.reshape(gray.shape)
        # binary = (labels * 255).astype("uint8")

    elif algo == "GrabCut 交互式":
        mask = np.zeros(img.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)

        if roi_mask is not None:
            rect = cv2.boundingRect(roi_mask)
            cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
            binary = (mask2 * 255).astype("uint8")
        else:
            binary = np.zeros_like(gray)

    return binary

# ========== 处理与显示 ==========
if uploaded_file is not None:
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    roi_mask = None
    if algo == "GrabCut 交互式":
        st.subheader("请在画布上绘制 ROI（矩形框）")
        canvas = st_canvas(
            fill_color="rgba(255,0,0,0.3)",
            stroke_color="red",
            stroke_width=2,
            background_image=img[:,:,::-1],
            update_streamlit=True,
            height=img.shape[0],
            width=img.shape[1],
            drawing_mode="rect",
            key="canvas",
        )
        if canvas.json_data is not None and len(canvas.json_data["objects"]) > 0:
            obj = canvas.json_data["objects"][0]
            x, y, w, h = int(obj["left"]), int(obj["top"]), int(obj["width"]), int(obj["height"])
            roi_mask = np.zeros(img.shape[:2], np.uint8)
            roi_mask[y:y+h, x:x+w] = 1

    # 二值化
    binary = binarize(img, algo, thresh_val, roi_mask)

    # 显示前后对比
    image_comparison(
        img1=img[:,:,::-1],  # BGR->RGB
        img2=cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB),
        label1="Before",
        label2="After"
    )

    # 满浆率计算
    total_pixels = binary.size
    black_pixels = np.sum(binary == 255)
    full_slurry_rate = black_pixels / total_pixels * 100

    st.markdown("---")
    st.subheader("结果指标")
    st.metric(label="满浆率 (%)", value=f"{full_slurry_rate:.2f}")
