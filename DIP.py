import cv2
import numpy as np

# -------------------------- 颜色HSV范围（保持稳定，确保掩码准确） --------------------------
color_hsv = {
    'pink':    [(3, 30, 120), (12, 80, 220)],   # 粉色保持正常
    'black':   [(0, 0, 30), (179, 80, 110)],    # 黑球包含反光
    'brown':   [(5, 60, 50), (18, 255, 160)],   # 棕球范围
    'blue':    [(90, 60, 50), (130, 255, 255)], # 蓝色保持正常
    'yellow':  [(20, 50, 60), (40, 255, 255)],  # 黄球范围
    'green':   [(25, 30, 40), (85, 255, 180)]   # 绿球保持可检测
}

# -------------------------- 强化棕球/黄球边缘平滑（核心优化） --------------------------
def detect_color_center_and_edge(img, color_name):
    lower, upper = color_hsv[color_name]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
    
    # 粉色处理（保持优化）
    if color_name == 'pink':
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # 绿球处理（保持可检测）
    elif color_name == 'green':
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # 黑球处理（保持可检测）
    elif color_name == 'black':
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # 棕球/黄球专用：强化平滑，消除褶皱
    elif color_name in ['brown', 'yellow']:
        # 1. 更大内核+更多闭操作：填充小褶皱，平滑边缘
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))  # 内核从5x5→7x7（覆盖更大褶皱）
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)  # 闭操作2次（深度填充小凹凸）
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)   # 轻度去噪，不破坏平滑轮廓
    
    # 蓝色处理（保持正常）
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # 轮廓检测
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f"警告：未检测到{color_name}轮廓，可能HSV范围仍需微调")
        return None, None
    
    # 面积阈值（棕球/黄球适当降低，适配平滑后的轮廓）
    min_area = {
        'pink': 100,
        'green': 80,
        'black': 80,
        'brown': 150,  # 棕球从200→150，适配平滑后的面积
        'blue': 200,
        'yellow': 120  # 黄球从150→120，适配平滑后的面积
    }[color_name]
    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    if not valid_contours:
        print(f"警告：{color_name}轮廓面积不足，已过滤")
        return None, None
    
    # 棕球/黄球轮廓平滑：多边形逼近（进一步消除微小褶皱）
    best_cnt = max(valid_contours, key=cv2.contourArea)
    if color_name in ['brown', 'yellow']:
        # 计算轮廓周长的1%作为逼近精度（值越小越接近原轮廓，稍大则更平滑）
        perimeter = cv2.arcLength(best_cnt, True)
        epsilon = 0.01 * perimeter  # 精度参数，0.01表示允许1%的误差
        best_cnt = cv2.approxPolyDP(best_cnt, epsilon, True)  # 多边形逼近，平滑轮廓
    
    # 中心计算（基于平滑后的轮廓）
    (cx, cy), _ = cv2.minEnclosingCircle(best_cnt)
    center = (int(cx), int(cy))
    
    return (color_name, center), best_cnt

# -------------------------- 图像处理与可视化 --------------------------
def process_image(img_path, save_path="final_result.png"):
    img = cv2.imread(img_path)
    if img is None:
        print("图片读取失败，请检查路径！")
        return
    
    result_img = img.copy()
    color_order = ['pink', 'black', 'brown', 'blue', 'yellow', 'green']
    result = []
    
    for color in color_order:
        color_info, contour = detect_color_center_and_edge(result_img, color)
        if color_info:
            result.append(color_info)
            color_name, (cx, cy) = color_info
            
            # 边缘线条厚度：棕球/黄球用3，确保平滑边缘清晰
            thickness = 4 if color_name == 'pink' else 3
            cv2.drawContours(result_img, [contour], -1, (0, 0, 0), thickness)
            
            # 绘制中心和标签
            circle_radius = 6 if color_name in ['black', 'green'] else 5
            cv2.circle(result_img, (cx, cy), circle_radius, (0, 0, 255), -1)
            cv2.putText(result_img, color_name, (cx + 10, cy), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    cv2.imwrite(save_path, result_img)
    cv2.imshow("Final Result (Brown/Yellow Smooth)", result_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("最终识别结果：", result)

# -------------------------- 执行 --------------------------
if __name__ == "__main__":
    process_image("test/1_1.png")  # 替换为实际图片路径