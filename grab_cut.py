import cv2
import numpy as np

# ========== Step 1: 读取图像和缩放 ==========
img = cv2.imread(r"C:\Users\Administrator\Desktop\001.bmp")
if img is None:
    raise FileNotFoundError("请检查图像路径！")

# 缩放图像以适应屏幕
def resize_image(image, max_width=1280, max_height=720):
    height, width = image.shape[:2]
    scale = min(max_width / width, max_height / height)
    if scale < 1:  # 只有当图像太大时才缩小
        new_width = int(width * scale)
        new_height = int(height * scale)
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
    return image

img = resize_image(img)

# 为 GrabCut 准备
mask = np.zeros(img.shape[:2], np.uint8)
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

rect = None   # 初始矩形区域
drawing = False
ix, iy = -1, -1

# ========== Step 2: 鼠标交互 ==========
def draw_rectangle(event, x, y, flags, param):
    global ix, iy, rect, drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        rect = (min(ix, x), min(iy, y), abs(ix-x), abs(iy-y))
        cv2.rectangle(display_img, (rect[0], rect[1]), 
                      (rect[0]+rect[2], rect[1]+rect[3]), (0,255,0), 2)

# ========== Step 3: 结果显示函数 ==========
def show_blend(val=50):
    # 获取滑动条位置比例
    slider_pos = val / 100.0
    h, w = img.shape[:2]
    
    # 创建左右分割显示
    split_pos = int(w * slider_pos)
    
    # 创建组合图像：左侧显示原图，右侧显示分割结果
    combined = img.copy()
    if binary_vis is not None:
        combined[:, split_pos:] = binary_vis[:, split_pos:]
    
    # 绘制分割线
    cv2.line(combined, (split_pos, 0), (split_pos, h), (0, 255, 0), 2)
    
    # 显示图像
    cv2.imshow("GrabCut Segmentation", combined)

# ========== Step 4: 主逻辑 ==========
display_img = img.copy()
result_vis = np.zeros_like(img)  # 初始化result_vis变量
binary_vis = None  # 初始化二值化可视化图像
cv2.namedWindow("GrabCut Segmentation")
cv2.setMouseCallback("GrabCut Segmentation", draw_rectangle)
cv2.createTrackbar("滑动分割线", "GrabCut Segmentation", 50, 100, show_blend)

while True:
    cv2.imshow("GrabCut Segmentation", display_img)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("r"):  # 按r键执行GrabCut
        # 执行 GrabCut
        if rect is not None:
            cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
            mask2 = np.where((mask==2)|(mask==0), 0, 1).astype("uint8")
            seg = img * mask2[:, :, np.newaxis]
            
            # 转二值可视化图（前景=白，背景=黑）
            binary = (mask2 * 255).astype("uint8")
            binary_vis = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            
            # 更新显示
            show_blend(cv2.getTrackbarPos("滑动分割线", "GrabCut Segmentation"))

    elif key == ord("b"):  # 按b键查看二值化效果
        if binary_vis is not None:
            cv2.imshow("二值化结果", binary_vis)
    
    elif key == 27:  # Esc 退出
        break

cv2.destroyAllWindows()