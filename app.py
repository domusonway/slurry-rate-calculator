# app.py
import io
import base64
import cv2
import numpy as np
import streamlit as st
from PIL import Image
import os
from datetime import datetime

# -------------------------
# 现在可以安全导入 canvas
# -------------------------
from streamlit_image_comparison import image_comparison
from streamlit_drawable_canvas import st_canvas

# 页面配置
st.set_page_config(layout="wide")

# 标题和文档入口
col1, col2 = st.columns([3, 1])
with col1:
    st.title("满浆率计算")
with col2:
    st.write("")  # 添加一些垂直空间
    if st.button("📖 使用手册", help="查看详细使用说明"):
        # 读取并显示用户手册
        try:
            with open("user_manual.md", "r", encoding="utf-8") as f:
                manual_content = f.read()
            
            # 在侧边栏显示手册
            with st.sidebar:
                st.markdown("---")
                st.markdown("## 📖 使用手册")
                st.markdown(manual_content)
        except FileNotFoundError:
            st.error("用户手册文件未找到")
        except Exception as e:
            st.error(f"读取手册失败: {e}")

# ========== 左侧控制面板 ==========
st.sidebar.header("输入")
uploaded_file = st.sidebar.file_uploader("上传图像", type=["jpg", "png", "bmp"])

algo = st.sidebar.selectbox(
    "选择二值化算法",
    ["Otsu", "全局阈值", "自适应阈值"]
    # ["全局阈值", "Otsu", "自适应阈值", "K-means 聚类", "GrabCut 交互式"]
)

thresh_val = st.sidebar.slider("阈值/参数调节", 0, 255, 160)

tile_type = st.sidebar.selectbox(
    "瓷砖类型",
    ["黑胶白砖", "白胶黑砖"]
)

# ========== 图像拍摄注意事项 ==========
st.sidebar.markdown("---")
st.sidebar.markdown("### 📸 图像拍摄注意事项")

with st.sidebar.expander("⚠️ 重要提示", expanded=True):
    st.markdown("""
    **拍摄要求：**
    
    1. **📐 保持平行**
       - 拍摄时与砖面尽可能平行
       - 避免倾斜角度影响精度
    
    2. **💡 光线充足**  
       - 拍摄光线保持明亮
       - 胶面无明显阴影
    
    3. **✂️ 图像裁剪**
       - 输入图像要做裁剪
       - 确保仅包含砖的部分
       - 不要包含任何背景
    """)

with st.sidebar.expander("💡 最佳实践"):
    st.markdown("""
    - **分辨率**：建议800x600以上
    - **格式**：推荐JPG格式
    - **对比度**：确保胶浆与瓷砖颜色对比明显
    - **清晰度**：避免模糊和抖动
    - **完整性**：确保砖面完整无遮挡
    """)

# 添加收款码图片到侧边栏最下方
st.sidebar.markdown("---")
st.sidebar.markdown("### 💝 支持开发")
try:
    qr_image_path = "img/赞赏码.jpg"
    if os.path.exists(qr_image_path):
        st.sidebar.image(qr_image_path, caption="如果这个工具对您有帮助，欢迎赞赏支持！", use_column_width=True)
    else:
        st.sidebar.info("💡 如果这个工具对您有帮助，欢迎支持开发！")
except Exception:
    st.sidebar.info("💡 如果这个工具对您有帮助，欢迎支持开发！")

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


# ========== 结果图生成函数 ==========
def create_result_image(original_img, binary_img, slurry_rate, filename):
    """
    创建结果图：原图和二值化图并排显示，并在图上绘制满浆率信息
    
    参数：
      original_img: 原始图像 (BGR)
      binary_img: 二值化图像 (单通道)
      slurry_rate: 满浆率百分比
      filename: 原始文件名
    
    返回：
      result_img: 组合后的结果图像 (BGR)
    """
    # 将二值化图转为三通道
    binary_bgr = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
    
    # 获取图像尺寸
    h, w = original_img.shape[:2]
    
    # 创建组合图像：左边原图，右边二值化图
    result_img = np.zeros((h, w * 2, 3), dtype=np.uint8)
    result_img[:, :w] = original_img
    result_img[:, w:] = binary_bgr
    
    # 在图像上绘制文本信息
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.8, min(w / 800, 2.0))  # 根据图像宽度调整字体大小
    thickness = max(1, int(font_scale * 2))
    
    # 文本内容
    text_lines = [
        f"File: {filename}",
        f"Slurry Rate: {slurry_rate:.2f}%",
        f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    ]
    
    # 计算文本位置（在图像底部）
    text_height = 30 * font_scale
    y_start = h - len(text_lines) * int(text_height) - 10
    
    # 绘制黑色背景矩形
    bg_height = len(text_lines) * int(text_height) + 20
    cv2.rectangle(result_img, (0, y_start - 10), (w * 2, h), (0, 0, 0), -1)
    
    # 绘制文本
    for i, text in enumerate(text_lines):
        y_pos = y_start + i * int(text_height)
        cv2.putText(result_img, text, (10, y_pos), font, font_scale, (255, 255, 255), thickness)
    
    # 在中间绘制分割线
    cv2.line(result_img, (w, 0), (w, h), (255, 255, 255), 2)
    
    # 添加标签
    cv2.putText(result_img, "Original", (10, 30), font, font_scale, (255, 255, 255), thickness)
    cv2.putText(result_img, "Binary", (w + 10, 30), font, font_scale, (255, 255, 255), thickness)
    
    return result_img


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

    # 生成结果图
    input_filename = uploaded_file.name
    result_img = create_result_image(img, binary, full_slurry_rate, input_filename)
    
    # 显示结果图
    st.subheader("结果图")
    st.image(result_img[:, :, ::-1], caption="原图与二值化结果对比", use_column_width=True)
    
    # 保存和下载功能
    col1, col2, col3 = st.columns([1, 1, 3])
    
    # 生成结果图数据
    input_filename = uploaded_file.name
    base_name = os.path.splitext(input_filename)[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_filename = f"{base_name}_result_{timestamp}.jpg"
    
    # 将结果图编码为字节数据
    _, buffer = cv2.imencode('.jpg', result_img)
    img_bytes = buffer.tobytes()
    
    with col1:
        # 直接下载按钮（推荐）
        st.download_button(
            label="📥 下载结果图",
            data=img_bytes,
            file_name=save_filename,
            mime="image/jpeg",
            type="primary",
            help="直接下载到浏览器默认下载文件夹"
        )
    
    with col2:
        # 保存到服务器temp目录的按钮
        if st.button("💾 保存到服务器", help="保存到应用服务器的temp目录"):
            save_path = os.path.join("temp", save_filename)
            os.makedirs("temp", exist_ok=True)
            
            success = cv2.imwrite(save_path, result_img)
            if success:
                st.success(f"✅ 已保存至服务器: {save_path}")
            else:
                st.error("❌ 保存失败，请检查文件路径权限")
    
    with col3:
        st.write("")  # 占位符

    # 可视化：把二值 mask 以半透明红色叠加到原图上
    try:
        overlay = img.copy()
        mask_bool = (binary == 255)
        overlay[mask_bool] = (0, 255, 0)  # 绿色 BGR
        blended = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)
        st.image(blended[:, :, ::-1], caption="满浆掩码叠加 (半透明红)", use_column_width=True)
        
        # 为叠加图添加下载按钮
        overlay_filename = f"overlay_{timestamp}.jpg"
        
        # 将叠加图编码为字节数据
        overlay_rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
        overlay_pil = Image.fromarray(overlay_rgb)
        overlay_buffer = io.BytesIO()
        overlay_pil.save(overlay_buffer, format='JPEG', quality=95)
        overlay_bytes = overlay_buffer.getvalue()
        
        # 创建两列布局
        col1_overlay, col2_overlay, col3_overlay = st.columns([1, 1, 2])
        
        with col1_overlay:
            # 直接下载按钮（推荐）
            st.download_button(
                label="📥 下载叠加图",
                data=overlay_bytes,
                file_name=overlay_filename,
                mime="image/jpeg",
                type="secondary",
                help="直接下载叠加图到浏览器默认下载文件夹"
            )
        
        with col2_overlay:
            # 保存到服务器temp目录的按钮
            if st.button("💾 保存叠加图到服务器", help="保存叠加图到应用服务器的temp目录"):
                save_path = os.path.join("temp", overlay_filename)
                os.makedirs("temp", exist_ok=True)
                
                success = cv2.imwrite(save_path, blended)
                if success:
                    st.success(f"✅ 叠加图已保存至服务器: {save_path}")
                else:
                    st.error("❌ 叠加图保存失败，请检查文件路径权限")
        
        with col3_overlay:
            st.write("")  # 占位符
            
    except Exception:
        pass

else:
    st.info("请在左侧上传图像以开始。")
