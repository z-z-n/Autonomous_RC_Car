# Author: ZZN
import cv2
import numpy as np

capture = cv2.VideoCapture('Input_Video.mp4')


def color_filter(image):
    # convert to HLS to mask based on HLS
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    lower = np.array([0, 200, 0])
    upper = np.array([255, 255, 255])
    yellower = np.array([10, 0, 90])
    yelupper = np.array([200, 255, 255])
    yellowmask = cv2.inRange(hls, yellower, yelupper)
    whitemask = cv2.inRange(hls, lower, upper)
    mask = cv2.bitwise_or(yellowmask, whitemask)
    masked = cv2.bitwise_and(image, image, mask=mask)
    return masked


def get_edge_img(color_img):
    HLS = color_filter(color_img)
    gray = cv2.cvtColor(HLS, cv2.COLOR_RGB2GRAY)
    img = cv2.resize(gray, (800, 450), interpolation=cv2.INTER_AREA)
    img0 = cv2.resize(color_img, (800, 450), interpolation=cv2.INTER_AREA)
    # Canny edge detection algorithm
    edge_img = cv2.Canny(img, 90, 200)
    return edge_img, img0


def roi_mask(edge_img):
    mask = np.zeros_like(edge_img)
    cv2.fillPoly(mask, np.array([[[180, 415], [300, 280], [450, 280], [720, 415]]]), color=255)
    masked_edge_img = cv2.bitwise_and(edge_img, mask)
    return masked_edge_img


'''3.霍夫变换，找出直线'''
'''3. Hough transform, find the straight line'''


def calculate_slope(line):
    '''计算线段line的斜率Calculate the slope of the line segment
    ：param Line：np.array([[x_1,y_1,x_2,y_2]])
    :return:
    '''
    x_1, y_1, x_2, y_2 = line[0]
    return (y_2 - y_1) / (x_2 - x_1)


'''4.离群值过滤'''
'''4. Outlier filtering'''


def reject_abnormal_lines(lines, threshold):
    '''剔出斜率不一致的线段'''
    slopes = [calculate_slope(line) for line in lines]
    while len(lines) > 0:
        mean = np.mean(slopes)
        diff = [abs(s - mean) for s in slopes]
        idx = np.argmax(diff)
        if diff[idx] > threshold:
            slopes.pop(idx)
            lines.pop(idx)
        else:
            break
    return lines


'''5.最小二乘拟合 把识别到的多条线段拟合成一条直线'''
'''5. Least square fitting Fit the identified multiple line segments into a straight line'''


# np.ravel: 将高维数组拉成一维数组
# np.polyfit:多项式拟合
# np.polyval: 多项式求值
def least_squares_fit(lines):
    x_coords = np.ravel([[line[0][0], line[0][2]] for line in lines])
    y_coords = np.ravel([[line[0][1], line[0][3]] for line in lines])  # 取出所有标点
    poly = np.polyfit(x_coords, y_coords, deg=1)  # 进行直线拟合，得到多项式系数
    point_min = (np.min(x_coords), np.polyval(poly, np.min(x_coords)))
    point_max = (np.max(x_coords), np.polyval(poly, np.max(x_coords)))  # 根据多项式系数，计算两个直线上的点
    return np.array([point_min, point_max], dtype=np.int64)


def get_lines(mask_gray_img):
    lines = cv2.HoughLinesP(mask_gray_img, 1, np.pi / 180, 15, minLineLength=30, maxLineGap=200)  # 获取所有线段
    # print('0', len(lines))
    left_lines = [line for line in lines if calculate_slope(line) > 0]
    right_lines = [line for line in lines if calculate_slope(line) < 0]
    # print('1',len(left_lines), len(right_lines))

    reject_abnormal_lines(left_lines, threshold=0.3)
    reject_abnormal_lines(right_lines, threshold=0.3)
    # print('2',len(left_lines), len(right_lines))
    left_lines = least_squares_fit(left_lines)
    right_lines = least_squares_fit(right_lines)
    return left_lines, right_lines


def calculate_x(line):
    x_1, y_1, x_2, y_2 = line[0]
    return [int(-(y_2 - 420) * (x_2 - x_1) / (y_2 - y_1) + x_2), 420]


def draw_lines(color_img, line1, line2):
    list1 = np.append(line1[0], line1[1])
    list2 = np.append(line2[0], line2[1])
    p1 = calculate_x([list1])
    p2 = calculate_x([list2])
    cv2.fillPoly(color_img, np.array([[[line1[0][0], line1[0][1]], p1,
                                       p2, [line2[1][0], line2[1][1]]]]), (100, 87, 249))
    cv2.line(color_img, tuple(line1[0]), tuple(p1), color=(0, 255, 0), thickness=2)
    cv2.line(color_img, tuple(p2), tuple(line2[1]), color=(255, 0, 0), thickness=2)
    if line1[0][1] > line2[1][1] and abs(line2[1][1] - line1[0][1]) > 5:
        cv2.putText(color_img, 'turn right', (0, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 5)
    elif line1[0][1] < line2[1][1] and abs(line2[1][1] - line1[0][1]) > 5:
        cv2.putText(color_img, 'turn left', (0, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 5)
    else:
        cv2.putText(color_img, 'go straight', (0, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 5)


def show_lane(color_img):
    """
    在 color_img 上画出车道线
    :param color_img：彩色图，channels=3
    :return:
    """
    edge_img, img0 = get_edge_img(color_img)
    mask_gray_img = roi_mask(edge_img)
    llines, rlines = get_lines(mask_gray_img)
    draw_lines(img0, llines, rlines)
    return img0, mask_gray_img
    # return mask_gray_img


fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out_cat = cv2.VideoWriter("save.mp4", fourcc, 15, (800, 450))  # 保存位置/格式
out1 = cv2.VideoWriter("gray.mp4", fourcc, 15, (800, 450), 0)
while True:
    ret, frame = capture.read()
    c = cv2.waitKey(10)
    if c == 27 or not ret:
        break
    frame1, frame2 = show_lane(frame)
    out_cat.write(frame1)  # 保存视频
    out1.write(frame2)
    # frame1 = show_lane(frame)
    cv2.imshow('frame', frame1)
    cv2.imshow('frame2', frame2)
out_cat.release()
out1.release()
