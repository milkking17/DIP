import cv2
import numpy as np
import os

# -------------------------- 颜色配置（含编号映射） --------------------------
color_hsv = {
    'pink':    [(3, 30, 120), (8, 80, 220)],   # 1 粉色
    'black':   [(0, 0, 30), (179, 50, 80)],    # 0 黑色
    'brown':   [(6, 60, 50), (18, 255, 160)],   # 3 棕色
    'blue':    [(90, 60, 50), (130, 255, 255)], # 2 蓝色
    'yellow':  [(20, 50, 60), (40, 255, 255)],  # 5 黄色
    'green':   [(25, 30, 40), (85, 255, 180)]   # 4 绿色
}

# 颜色到编号的映射（0黑/1粉/2蓝/3棕/4绿/5黄）
color_to_id = {
    'black': 0,
    'pink': 1,
    'blue': 2,
    'brown': 3,
    'green': 4,
    'yellow': 5
}

# 按输出顺序处理颜色（确保单图内顺序：黑、粉、蓝、棕、绿、黄）
color_order_output = ['black', 'pink', 'blue', 'brown', 'green', 'yellow']
# 原识别顺序（保持原有识别逻辑）
color_order_recognize = ['pink', 'black', 'brown', 'blue', 'yellow', 'green']

# -------------------------- 图像处理主逻辑 --------------------------
def process_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print("无法读取图片:", img_path)
        return [], []
    
    result_img = img.copy()
    result_original = []  # 原格式结果
    result_formatted = [] # 指定格式结果
    color_positions = {}  # 存储颜色对应的坐标
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    for color in color_order_recognize:
        lower, upper = color_hsv[color]
        mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
        
        # 形态学预处理
        if color == 'pink':
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        elif color in ['brown', 'yellow']:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        else:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # 高斯模糊+Canny边缘检测
        blurred = cv2.GaussianBlur(img, (5, 5), 0)
        masked_img = cv2.bitwise_and(blurred, blurred, mask=mask)
        gray = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)
        
        # 差异化Canny阈值
        if color == 'black':
            edges = cv2.Canny(gray, 20, 60)
        elif color in ['pink', 'green']:
            edges = cv2.Canny(gray, 50, 150)
        else:
            edges = cv2.Canny(gray, 80, 200)
        
        # 边缘优化
        edge_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.morphologyEx(edges, cv2.MORPH_DILATE, edge_kernel, iterations=1)
        
        # 轮廓检测与筛选
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            min_area = {
                'pink': 100, 'green': 80, 'black': 80,
                'brown': 150, 'blue': 200, 'yellow': 120
            }[color]
            valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
            
            if valid_contours:
                best_cnt = max(valid_contours, key=cv2.contourArea)
                
                # 中心计算
                (cx, cy), _ = cv2.minEnclosingCircle(best_cnt)
                center = (int(cx), int(cy))
                result_original.append((color, center))
                color_positions[color] = (int(cx), int(cy))
                
                # 绘制轮廓和中心
                cv2.drawContours(result_img, [best_cnt], -1, (0, 0, 0), 3)
                cv2.circle(result_img, center, 5, (0, 0, 255), -1)
                cv2.putText(result_img, color, (center[0] + 10, center[1]), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    # 按指定顺序生成格式结果
    for color in color_order_output:
        if color in color_positions:
            x, y = color_positions[color]
            result_formatted.append([x, y, color_to_id[color]])
    
    # 保存到当前目录，添加processed前缀
    filename = os.path.basename(img_path)
    name, ext = os.path.splitext(filename)
    save_path = f"processed_{name}{ext}"
    cv2.imwrite(save_path, result_img)
    
    return result_original, result_formatted

# -------------------------- 遍历test文件夹处理 --------------------------
if __name__ == "__main__":
    # 支持的图像格式
    supported_formats = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')
    
    # 存储所有图片的指定格式结果
    all_formatted_results = []
    
    # 遍历test文件夹
    if os.path.exists("test"):
        for filename in os.listdir("test"):
            if filename.lower().endswith(supported_formats):
                img_path = os.path.join("test", filename)
                _, formatted_result = process_image(img_path)
                all_formatted_results.extend(formatted_result)
    else:
        print("test文件夹不存在")
    
    # 输出指定格式结果（无换行）
    print(str(all_formatted_results).replace('\n', ''))
    
    cv2.destroyAllWindows()