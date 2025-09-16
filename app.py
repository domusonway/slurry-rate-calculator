# app.py
import io
import base64
import cv2
import numpy as np
import streamlit as st
from PIL import Image

# -------------------------
# 现在可以安全导入 canvas
# -------------------------
from streamlit_image_comparison import image_comparison
from streamlit_drawable_canvas import st_canvas

# 页面配置
st.set_page_config(layout="wide")
st.title("满浆率计算")

# ========== 左侧控制面板 ==========
st.sidebar.header("输入")
uploaded_file = st.sidebar.file_uploader("上传图像", type=["jpg", "png", "bmp"])

algo = st.sidebar.selectbox(
    "选择二值化算法",
    ["全局阈值", "Otsu", "自适应阈值", "K-means 聚类", "GrabCut 交互式"]
)

thresh_val = st.sidebar.slider("阈值/参数调节", 0, 255, 160)

tile_type = st.sidebar.selectbox(
    "瓷砖类型",
    ["黑胶白砖", "白胶黑砖"]
)

# ========== 二值化函数 ==========
def binarize(img, algo, val, roi_mask=None):
    """
    输入：
      img: OpenCV BGR image (np.ndarray)
      algo: 算法名
      val: 阈值/参数
      roi_mask: GrabCut 的 ROI mask (0/1)
    返回：
      binary: 单通道 uint8 二值图 (0 或 255)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 默认兜底
    binary = np.zeros_like(gray)

    if algo == "全局阈值":
        _, binary = cv2.threshold(gray, val, 255, cv2.THRESH_BINARY)

    elif algo == "Otsu":
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    elif algo == "自适应阈值":
        # blockSize 必须为奇数且 >=3
        bs = val if (val % 2 == 1 and val >= 3) else max(3, (val // 2) * 2 + 1)
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                       cv2.THRESH_BINARY, blockSize=bs, C=2)

    elif algo == "K-means 聚类":
        st.info("K-means 聚类算法需要安装 sklearn，已跳过。")
        # try:
        #     Z = gray.reshape((-1, 1)).astype(np.float32)
        #     from sklearn.cluster import KMeans  # 可能没有安装 sklearn
        #     kmeans = KMeans(n_clusters=2, n_init=10).fit(Z)
        #     labels = kmeans.labels_.reshape(gray.shape)
        #     # 把聚类标签映射为 0/255；为了可读性把类 1 映为 255
        #     binary = (labels.astype(np.uint8) * 255)
        # except Exception as e:
        #     st.warning("K-means 聚类失败（可能未安装 sklearn），已返回全黑图像。")
        #     binary = np.zeros_like(gray)

    elif algo == "GrabCut 交互式":
        mask = np.zeros(img.shape[:2], np.uint8)
        bgdModel = np.zeros((1, 65), np.float64)
        fgdModel = np.zeros((1, 65), np.float64)

        if roi_mask is not None:
            ys, xs = np.where(roi_mask == 1)
            if len(xs) > 0 and len(ys) > 0:
                # rect 格式为 (x, y, w, h)
                rect = (int(xs.min()), int(ys.min()), int(xs.max() - xs.min()), int(ys.max() - ys.min()))
                # 为防止 rect 非法，做最小值校验
                if rect[2] <= 0 or rect[3] <= 0:
                    binary = np.zeros_like(gray)
                else:
                    try:
                        cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
                        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
                        binary = (mask2 * 255).astype("uint8")
                    except Exception as e:
                        st.warning(f"GrabCut 运行失败: {e}")
                        binary = np.zeros_like(gray)
            else:
                binary = np.zeros_like(gray)
        else:
            binary = np.zeros_like(gray)

    return binary


# ========== 处理与显示 ==========
if uploaded_file is not None:
    # 读取图像
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if img is None:
        st.error("无法解码上传的图像，请确认文件完整且为 jpg/png/bmp。")
        st.stop()

    roi_mask = None
    if algo == "GrabCut 交互式":
        st.subheader("请在画布上绘制 ROI（矩形框）")

        # 转为 PIL.Image（RGB）传给 canvas；canvas 内部会调用我们 monkey-patch 的 image_to_url
        bg_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        canvas = st_canvas(
            fill_color="rgba(255,0,0,0.3)",
            stroke_color="red",
            stroke_width=2,
            background_image=bg_img,
            update_streamlit=True,
            height=img.shape[0],
            width=img.shape[1],
            drawing_mode="rect",
            key="canvas",
        )

        # st_canvas 可能返回 None（取决于组件版本），先做健壮性检查
        if canvas is not None and getattr(canvas, "json_data", None) is not None:
            objects = canvas.json_data.get("objects", [])
            if len(objects) > 0:
                obj = objects[0]
                # 注意：fabric.js 的 left/top/width/height 可能为浮点，转换为 int
                x, y, w, h = int(obj.get("left", 0)), int(obj.get("top", 0)), int(obj.get("width", 0)), int(obj.get("height", 0))
                # 防越界
                x = max(0, x); y = max(0, y)
                w = max(0, min(w, img.shape[1] - x))
                h = max(0, min(h, img.shape[0] - y))
                roi_mask = np.zeros(img.shape[:2], np.uint8)
                roi_mask[y:y+h, x:x+w] = 1

    # 二值化
    binary = binarize(img, algo, thresh_val, roi_mask)

    # 满浆率计算（白色像素视作满浆）
    if tile_type == "黑胶白砖":
        # 业务：白胶黑砖时，先反转，使满浆部分变为白色（255）
        binary = cv2.bitwise_not(binary)

    # 显示前后对比（streamlit_image_comparison 需要安装）
    try:
        image_comparison(
            img1=img[:, :, ::-1],  # BGR->RGB
            img2=cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB),
            label1="Before",
            label2="After"
        )
    except Exception:
        # 退回到简单显示
        st.image(img[:, :, ::-1], caption="Before", use_column_width=True)
        st.image(cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB), caption="After", use_column_width=True)

    total_pixels = binary.size
    if total_pixels == 0:
        st.error("图像尺寸异常，无法计算。")
        st.stop()

    slurry_pixels = int(np.sum(binary == 255))
    full_slurry_rate = slurry_pixels / total_pixels * 100

    st.markdown("---")
    st.subheader("结果指标")
    st.metric(label="满浆率 (%)", value=f"{full_slurry_rate:.2f}")

    # 可视化：把二值 mask 以半透明红色叠加到原图上
    try:
        overlay = img.copy()
        mask_bool = (binary == 255)
        overlay[mask_bool] = (0, 255, 0)  # 绿色 BGR
        blended = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)
        st.image(blended[:, :, ::-1], caption="满浆掩码叠加 (半透明红)", use_column_width=True)
    except Exception:
        pass

else:
    st.info("请在左侧上传图像以开始。")
