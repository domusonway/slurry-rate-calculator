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
input_mode = st.sidebar.radio("输入模式", ["单文件", "文件夹"], index=0)
uploaded_file = None
batch_images = None
batch_zip = None
start_batch = False

# 批处理结果持久化（避免点击下载导致重渲染后按钮消失）
if "batch_results" not in st.session_state:
    st.session_state["batch_results"] = None
if "batch_saved_files" not in st.session_state:
    st.session_state["batch_saved_files"] = None
if "batch_timestamp" not in st.session_state:
    st.session_state["batch_timestamp"] = None
if "batch_output_dir" not in st.session_state:
    st.session_state["batch_output_dir"] = None

if input_mode == "单文件":
    uploaded_file = st.sidebar.file_uploader("上传图像", type=["jpg", "jpeg", "png", "bmp"])
else:
    batch_images = st.sidebar.file_uploader(
        "批量上传图像",
        type=["jpg", "jpeg", "png", "bmp"],
        accept_multiple_files=True,
        help="选择或拖拽多张图片进行批处理"
    )
    batch_zip = st.sidebar.file_uploader(
        "或上传 ZIP 压缩包",
        type=["zip"],
        help="上传包含图片的ZIP压缩包"
    )
    start_batch = st.sidebar.button("开始批处理")

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

# 添加测试项描述输入框
test_description = st.sidebar.text_input(
    "测试项描述",
    value="满浆率检测",
    help="输入测试项目的描述信息，将显示在结果图像上"
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
        st.sidebar.image(qr_image_path, caption="如果这个工具对您有帮助，欢迎赞赏支持！", width="stretch")
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
def create_result_image(original_img, binary_img, slurry_rate, filename, test_description="满浆率检测"):
    """
    创建结果图：原图和二值化图并排显示，并在图上绘制满浆率信息
    
    参数：
      original_img: 原始图像 (BGR)
      binary_img: 二值化图像 (单通道)
      slurry_rate: 满浆率百分比
      filename: 原始文件名
      test_description: 测试项描述
    
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
    
    # 使用PIL绘制支持中文的文本信息
    from PIL import Image, ImageDraw, ImageFont
    
    # 将OpenCV图像转换为PIL图像
    pil_img = Image.fromarray(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    
    # 设置字体大小 - 使用固定基准大小，不依赖图像尺寸
    base_font_size = 32  # 基准字体大小
    # 可选：根据图像大小进行适度调整，但设置更合理的范围
    font_size = max(28, min(base_font_size + (w - 800) // 100, 48))
    
    # 跨平台字体加载
    import platform
    import os
    
    font = None
    system = platform.system()
    
    # 定义不同系统的字体路径和字体文件
    font_paths = []
    if system == "Windows":
        font_paths = [
            "C:/Windows/Fonts/msyh.ttc",  # 微软雅黑
            "C:/Windows/Fonts/simsun.ttc",  # 宋体
            "C:/Windows/Fonts/arial.ttf",  # Arial
        ]
    elif system == "Darwin":  # macOS
        font_paths = [
            "/System/Library/Fonts/PingFang.ttc",  # 苹方
            "/System/Library/Fonts/Arial.ttf",  # Arial
            "/System/Library/Fonts/Helvetica.ttc",  # Helvetica
        ]
    else:  # Linux (包括云端部署环境)
        font_paths = [
            # 常见的Linux字体路径
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # DejaVu Sans
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",  # Liberation Sans
            "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",  # Noto Sans CJK
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",  # Noto Sans CJK (备用路径)
            "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",  # Ubuntu字体
            "/usr/share/fonts/truetype/droid/DroidSansFallback.ttf",  # Droid Sans Fallback
            "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",  # 文泉驿微米黑
            "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",  # 文泉驿正黑
            # 云端部署环境可能的字体路径
            "/app/.fonts/NotoSansCJK-Regular.ttc",  # 自定义字体目录
            "/tmp/fonts/NotoSansCJK-Regular.ttc",  # 临时字体目录
            # 更多备用选项
            "/usr/share/fonts/TTF/DejaVuSans.ttf",
            "/usr/share/fonts/TTF/LiberationSans-Regular.ttf",
            "/System/Library/Fonts/Arial.ttf",  # 某些Linux发行版可能有
        ]
    
    # 尝试加载字体
    for font_path in font_paths:
        try:
            if os.path.exists(font_path):
                font = ImageFont.truetype(font_path, font_size)
                break
        except Exception:
            continue
    
    # 如果所有字体都加载失败，使用默认字体
    if font is None:
        font = ImageFont.load_default()
    
    # 文本内容
    text_lines = [
        f"Test: {test_description}",
        f"File: {filename}",
        f"Slurry Rate: {slurry_rate:.2f}%",
        f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    ]
    
    # 计算文本位置（在图像底部）
    line_height = font_size + 5
    y_start = h - len(text_lines) * line_height - 10
    
    # 绘制黑色背景矩形
    bg_height = len(text_lines) * line_height + 10
    # draw.rectangle([(0, y_start - 5), (w * 2, h)], fill=(0, 0, 0, 180))
    
    # 绘制文本
    for i, text in enumerate(text_lines):
        y_pos = y_start + i * line_height
        draw.text((10, y_pos), text, font=font, fill=(255, 255, 255))
    
    # 将PIL图像转换回OpenCV格式
    result_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    
    # 在中间绘制分割线
    cv2.line(result_img, (w, 0), (w, h), (255, 255, 255), 2)
    
    # 使用PIL添加标签
    pil_img_labels = Image.fromarray(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    draw_labels = ImageDraw.Draw(pil_img_labels)
    
    # 设置标签字体 - 使用固定基准大小
    base_label_font_size = 36  # 标签基准字体大小
    label_font_size = max(32, min(base_label_font_size + (w - 800) // 80, 52))
    
    # 跨平台标签字体加载
    label_font = None
    
    # 尝试加载字体
    for font_path in font_paths:
        try:
            if os.path.exists(font_path):
                label_font = ImageFont.truetype(font_path, label_font_size)
                break
        except Exception:
            continue
    
    # 如果所有字体都加载失败，使用默认字体
    if label_font is None:
        label_font = ImageFont.load_default()
    
    # 添加标签
    draw_labels.text((10, 10), "Original", font=label_font, fill=(255, 0, 0))
    draw_labels.text((w + 10, 10), "Binary", font=label_font, fill=(0, 255, 0))
    
    # 转换回OpenCV格式
    result_img = cv2.cvtColor(np.array(pil_img_labels), cv2.COLOR_RGB2BGR)

    return result_img


# ========== 处理与显示 ==========
if input_mode == "单文件":
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
            st.image(img[:, :, ::-1], caption="Before", width="stretch")
            st.image(cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB), caption="After", width="stretch")

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
        result_img = create_result_image(img, binary, full_slurry_rate, input_filename, test_description)
        
        # 显示结果图
        st.subheader("结果图")
        st.image(result_img[:, :, ::-1], caption="原图与二值化结果对比", width="stretch")
        
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
            
            # 在叠加图上绘制文本信息（使用PIL支持中文）
            h, w = blended.shape[:2]
            
            # 使用PIL绘制支持中文的文本信息
            from PIL import Image, ImageDraw, ImageFont
            
            # 将OpenCV图像转换为PIL图像
            pil_img = Image.fromarray(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(pil_img)
            
            # 设置字体大小 - 使用固定基准大小，不依赖图像尺寸
            base_font_size = 32  # 基准字体大小
            # 可选：根据图像大小进行适度调整，但设置更合理的范围
            font_size = max(28, min(base_font_size + (w - 800) // 100, 48))
            
            # 跨平台叠加图字体加载
            overlay_font = None
            
            # 重新定义字体路径（因为之前的font_paths在函数作用域外）
            import platform
            system = platform.system()
            
            # 定义不同系统的字体路径和字体文件
            overlay_font_paths = []
            if system == "Windows":
                overlay_font_paths = [
                    "C:/Windows/Fonts/msyh.ttc",  # 微软雅黑
                    "C:/Windows/Fonts/simsun.ttc",  # 宋体
                    "C:/Windows/Fonts/arial.ttf",  # Arial
                ]
            elif system == "Darwin":  # macOS
                overlay_font_paths = [
                    "/System/Library/Fonts/PingFang.ttc",  # 苹方
                    "/System/Library/Fonts/Arial.ttf",  # Arial
                    "/System/Library/Fonts/Helvetica.ttc",  # Helvetica
                ]
            else:  # Linux (包括云端部署环境)
                overlay_font_paths = [
                    # 常见的Linux字体路径
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # DejaVu Sans
                    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",  # Liberation Sans
                    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",  # Noto Sans CJK
                    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",  # Noto Sans CJK (备用路径)
                    "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",  # Ubuntu字体
                    "/usr/share/fonts/truetype/droid/DroidSansFallback.ttf",  # Droid Sans Fallback
                    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",  # 文泉驿微米黑
                    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",  # 文泉驿正黑
                    # 云端部署环境可能的字体路径
                    "/app/.fonts/NotoSansCJK-Regular.ttc",  # 自定义字体目录
                    "/tmp/fonts/NotoSansCJK-Regular.ttc",  # 临时字体目录
                    # 更多备用选项
                    "/usr/share/fonts/TTF/DejaVuSans.ttf",
                    "/usr/share/fonts/TTF/LiberationSans-Regular.ttf",
                    "/System/Library/Fonts/Arial.ttf",  # 某些Linux发行版可能有
                ]
            
            # 尝试加载字体
            for font_path in overlay_font_paths:
                try:
                    if os.path.exists(font_path):
                        overlay_font = ImageFont.truetype(font_path, font_size)
                        break
                except Exception:
                    continue
            
            # 如果所有字体都加载失败，使用默认字体
            if overlay_font is None:
                overlay_font = ImageFont.load_default()
            
            font = overlay_font  # 保持原变量名兼容性
            
            # 文本内容
            text_lines = [
                f"Test: {test_description}",
                f"File: {uploaded_file.name}",
                f"Slurry Rate: {full_slurry_rate:.2f}%",
                f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
            ]
            
            # 计算文本位置（在图像底部）
            line_height = font_size + 5
            y_start = h - len(text_lines) * line_height - 10
            
            # 绘制黑色背景矩形
            bg_height = len(text_lines) * line_height + 10
            # draw.rectangle([(0, y_start - 5), (w, h)], fill=(0, 0, 0, 180))
            
            # 绘制文本
            for i, text in enumerate(text_lines):
                y_pos = y_start + i * line_height
                draw.text((10, y_pos), text, font=font, fill=(255, 255, 255))
            
            # 将PIL图像转换回OpenCV格式
            blended = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            
            st.image(blended[:, :, ::-1], caption="满浆掩码叠加 (半透明红)", width="stretch")
            
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
                if st.button("💾 保存到服务器", help="保存叠加图到应用服务器的temp目录"):
                    save_path = os.path.join("temp", overlay_filename)
                    os.makedirs("temp", exist_ok=True)
                    
                    success = cv2.imwrite(save_path, blended)
                    if success:
                        st.success(f"✅ 叠加图已保存至服务器: {save_path}")
                    else:
                        st.error("❌ 叠加图保存失败，请检查文件路径权限")
            
            with col3_overlay:
                st.write("")  # 占位符
                
        except Exception as e:
            st.error(f"❌ 叠加图生成失败: {str(e)}")
            st.error("请检查图像处理过程中是否出现错误")
    else:
        st.info("请在左侧上传图像以开始。")

else:
    # 批量上传（多文件或ZIP），适用于部署环境，无需服务器路径
    if not batch_images and not batch_zip and st.session_state["batch_results"] is None:
        st.info("请上传多张图片或ZIP压缩包后点击“开始批处理”。")
    elif not start_batch and st.session_state["batch_results"] is None:
        if batch_images:
            st.info(f"检测到 {len(batch_images)} 个待处理图像。点击左侧“开始批处理”。")
        elif batch_zip:
            st.info(f"已选择 ZIP：{batch_zip.name}。点击左侧“开始批处理”。")
    elif start_batch:
        # 收集待处理图像
        valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
        items = []  # [(filename, img)]
        if batch_images:
            for f in batch_images:
                try:
                    data = f.read()
                    arr = np.frombuffer(data, np.uint8)
                    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if img is not None:
                        items.append((f.name, img))
                except Exception:
                    continue
        elif batch_zip:
            try:
                import zipfile
                zf = zipfile.ZipFile(io.BytesIO(batch_zip.read()))
                names = [n for n in zf.namelist() if os.path.splitext(n)[1].lower() in valid_exts]
                for n in names:
                    try:
                        data = zf.read(n)
                        arr = np.frombuffer(data, np.uint8)
                        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                        if img is not None:
                            items.append((os.path.basename(n), img))
                    except Exception:
                        continue
            except Exception as e:
                st.error(f"ZIP读取失败：{e}")

        if len(items) == 0:
            st.warning("未找到可处理图像。请确认上传的文件为JPG/PNG/BMP格式。")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join("temp", f"batch_{timestamp}")
            os.makedirs(output_dir, exist_ok=True)

            st.subheader("批处理进度")
            progress = st.progress(0)
            status = st.empty()

            results = []
            saved_files = []

            for idx, (fname, img) in enumerate(items, start=1):
                if img is None:
                    results.append({"文件名": fname, "状态": "读取失败", "满浆率(%)": None})
                    status.warning(f"[{idx}/{len(items)}] 读取失败：{fname}")
                    progress.progress(idx/len(items))
                    continue

                binary = binarize(img, algo, thresh_val, roi_mask=None)

                if tile_type == "黑胶白砖":
                    binary = cv2.bitwise_not(binary)

                total_pixels = binary.size
                if total_pixels == 0:
                    results.append({"文件名": fname, "状态": "尺寸异常", "满浆率(%)": None})
                    status.error(f"[{idx}/{len(items)}] 尺寸异常：{fname}")
                    progress.progress(idx/len(items))
                    continue

                slurry_pixels = int(np.sum(binary == 255))
                rate = slurry_pixels / total_pixels * 100.0

                result_img = create_result_image(img, binary, rate, fname, test_description)

                base_name = os.path.splitext(fname)[0]
                save_filename = f"{base_name}_result_{timestamp}.jpg"
                save_path = os.path.join(output_dir, save_filename)
                cv2.imwrite(save_path, result_img)
                saved_files.append(save_path)

                # 叠加图保存
                try:
                    overlay = img.copy()
                    mask_bool = (binary == 255)
                    overlay[mask_bool] = (0, 255, 0)
                    blended = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)
                    overlay_name = f"{base_name}_overlay_{timestamp}.jpg"
                    overlay_path = os.path.join(output_dir, overlay_name)
                    cv2.imwrite(overlay_path, blended)
                    saved_files.append(overlay_path)
                except Exception:
                    pass

                results.append({"文件名": fname, "状态": "完成", "满浆率(%)": f"{rate:.2f}"})
                status.info(f"[{idx}/{len(items)}] 完成：{fname}（满浆率 {rate:.2f}%）")
                progress.progress(idx/len(items))

            st.success(f"批处理完成，共处理 {len(items)} 张。输出目录：{output_dir}")
            # 将结果保存在会话状态，避免下载按钮触发重渲染后丢失
            st.session_state["batch_results"] = results
            st.session_state["batch_saved_files"] = saved_files
            st.session_state["batch_timestamp"] = timestamp
            st.session_state["batch_output_dir"] = output_dir

            # 渲染结果（首次）
            st.subheader("批处理结果")
            st.dataframe(results, use_container_width=True)

            csv_lines = ["文件名,满浆率(%)"]
            for r in results:
                if r["满浆率(%)"] is not None:
                    csv_lines.append(f"{r['文件名']},{r['满浆率(%)']}")
            csv_bytes = ("\n".join(csv_lines)).encode("utf-8-sig")
            st.download_button(
                label="📄 下载结果CSV",
                data=csv_bytes,
                file_name=f"batch_results_{timestamp}.csv",
                mime="text/csv",
            )

            zip_buffer = io.BytesIO()
            import zipfile as _zipfile
            with _zipfile.ZipFile(zip_buffer, "w", _zipfile.ZIP_DEFLATED) as zf:
                for p in saved_files:
                    zf.write(p, os.path.basename(p))
            zip_bytes = zip_buffer.getvalue()
            st.download_button(
                label="📦 下载所有结果ZIP",
                data=zip_bytes,
                file_name=f"batch_outputs_{timestamp}.zip",
                mime="application/zip",
                type="primary",
                help="包含所有结果图与叠加图"
            )

            # 提供重置按钮以清空会话状态
            if st.button("🧹 清空批处理结果"):
                st.session_state["batch_results"] = None
                st.session_state["batch_saved_files"] = None
                st.session_state["batch_timestamp"] = None
                st.session_state["batch_output_dir"] = None
                st.experimental_rerun()

    else:
        # 未再次点击开始，但已有批处理结果：继续展示下载按钮，避免消失
        if st.session_state["batch_results"]:
            results = st.session_state["batch_results"]
            saved_files = st.session_state["batch_saved_files"] or []
            timestamp = st.session_state["batch_timestamp"] or datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = st.session_state["batch_output_dir"] or "temp"

            st.success(f"批处理结果已缓存（输出目录：{output_dir}）")
            st.subheader("批处理结果")
            st.dataframe(results, use_container_width=True)

            csv_lines = ["文件名,满浆率(%)"]
            for r in results:
                if r["满浆率(%)"] is not None:
                    csv_lines.append(f"{r['文件名']},{r['满浆率(%)']}")
            csv_bytes = ("\n".join(csv_lines)).encode("utf-8-sig")
            st.download_button(
                label="📄 下载结果CSV",
                data=csv_bytes,
                file_name=f"batch_results_{timestamp}.csv",
                mime="text/csv",
            )

            zip_buffer = io.BytesIO()
            import zipfile as _zipfile
            with _zipfile.ZipFile(zip_buffer, "w", _zipfile.ZIP_DEFLATED) as zf:
                for p in saved_files:
                    try:
                        zf.write(p, os.path.basename(p))
                    except Exception:
                        continue
            zip_bytes = zip_buffer.getvalue()
            st.download_button(
                label="📦 下载所有结果ZIP",
                data=zip_bytes,
                file_name=f"batch_outputs_{timestamp}.zip",
                mime="application/zip",
                type="primary",
                help="包含所有结果图与叠加图"
            )

            if st.button("🧹 清空批处理结果"):
                st.session_state["batch_results"] = None
                st.session_state["batch_saved_files"] = None
                st.session_state["batch_timestamp"] = None
                st.session_state["batch_output_dir"] = None
                st.experimental_rerun()
