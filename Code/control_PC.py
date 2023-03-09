# PC作为服务器处理树莓派数据
import numpy as np
import cv2
import socket
import time
from collections import *

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
    # return regions with yellow and write color
    return masked


'''
2. Canny to get edges
'''


# canny to get edges of imgs
def get_edge_img(color_img):
    # get HLS img
    HLS = color_filter(color_img)
    # turn to gray
    gray = cv2.cvtColor(HLS, cv2.COLOR_RGB2GRAY)
    # resize to (800,450)
    img = cv2.resize(gray, (800, 450), interpolation=cv2.INTER_AREA)
    # resize original img
    img0 = cv2.resize(color_img, (800, 450), interpolation=cv2.INTER_AREA)
    # Canny edge detection algorithm
    edge_img = cv2.Canny(img, 90, 200)
    return edge_img, img0


'''
3. reverse regions of interest (For images of size 800×450)
'''


# region of interest
def roi_mask(edge_img):
    mask = np.zeros_like(edge_img)
    cv2.fillPoly(mask, np.array([[[180, 415], [300, 280], [450, 280], [720, 415]]]), color=255)
    masked_edge_img = cv2.bitwise_and(edge_img, mask)
    return masked_edge_img


'''
4.霍夫变换，找出直线 (Hough transform, find the straight line)
'''

'''
5.离群值过滤(Outlier filtering) 
'''


def reject_abnormal_lines(lines, threshold):
    '''剔出斜率不一致的线段(Pick out the line segments with inconsistent slopes)'''
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


'''
6.最小二乘拟合 把识别到的多条线段拟合成一条直线
(Least square fitting Fit the identified multiple line segments into a straight line)
'''
def calculate_slope(line):
    '''计算线段line的斜率Calculate the slope of the line segment
    ：param Line：np.array([[x_1,y_1,x_2,y_2]])
    :return:
    '''
    x_1, y_1, x_2, y_2 = line[0]
    return (y_2 - y_1) / (x_2 - x_1)

# np.ravel: 将高维数组拉成一维数组(pull a high-dimensional array into a one-dimensional array)
# np.polyfit:多项式拟合(polynomial fit)
# np.polyval: 多项式求值(polynomial valuation)
def least_squares_fit(lines):
    # get x,y of two points
    x_coords = np.ravel([[line[0][0], line[0][2]] for line in lines])
    y_coords = np.ravel([[line[0][1], line[0][3]] for line in lines])
    # 进行直线拟合，得到多项式系数（Perform a straight line fit to get the polynomial coefficients）
    poly = np.polyfit(x_coords, y_coords, deg=1)
    # 根据多项式系数，计算两个直线上的点 (Calculate the points on the two lines according to the polynomial coefficients)
    point_min = (np.min(x_coords), np.polyval(poly, np.min(x_coords)))
    point_max = (np.max(x_coords), np.polyval(poly, np.max(x_coords)))
    return np.array([point_min, point_max], dtype=np.int64)

# get one left line and one right line
def get_lines(mask_gray_img):
    # 获取所有线段 get all lines
    lines = cv2.HoughLinesP(mask_gray_img, 1, np.pi / 180, 15, minLineLength=30, maxLineGap=200)
    # 斜率分类 classification by slopes
    left_lines = [line for line in lines if calculate_slope(line) < 0]
    right_lines = [line for line in lines if calculate_slope(line) > 0]
    # print('1',len(left_lines), len(right_lines))
    # 删除离群线段 remove outlier lines by slopes
    reject_abnormal_lines(left_lines, threshold=0.3)
    reject_abnormal_lines(right_lines, threshold=0.3)
    # print('2',len(left_lines), len(right_lines))
    left_lines = least_squares_fit(left_lines)
    right_lines = least_squares_fit(right_lines)
    return left_lines, right_lines


# Calculate the x-coordinate of the point where the lane line intersects with y=420
# 计算车道线与y=420相交点的x坐标
def calculate_x(line):
    x_1, y_1, x_2, y_2 = line[0]
    return [int(-(y_2 - 420) * (x_2 - x_1) / (y_2 - y_1) + x_2), 420]


'''8.其他信息处理'''
# 透视变换
def perspective_img(img):
    # 图像
    # 图片大小，opencv读取，[1]是宽度[0]是高度
    img_size = (img.shape[1], img.shape[0])
    dst_size = (800, 450)
    # src：源图像中待测矩形的四点坐标
    # dst：目标图像中矩形的四点坐标
    dst0 = np.array([(0.2, 0), (0.8, 0), (0.2, 1), (0.8, 1)], dtype="float32")
    src0 = np.array([(0.44, 0.65), (0.57, 0.65), (0.1, 1), (1, 1)], dtype="float32")
    src = np.float32(img_size) * src0
    dst = np.float32(dst_size) * dst0
    R = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, R, dst_size)
    return warped

def direction(grayl,grayr):
    nonzerol = grayl.nonzero()
    nonzeror = grayr.nonzero()
    l_x = np.array(nonzerol[1])
    r_x = np.array(nonzeror[1])
    x_final = np.mean([l_x[0], r_x[0]])
    x_init = np.mean([l_x[-1], r_x[-1]])
    degree = (x_final - x_init) / x_init
    avg_degree = frame_img.update(degree)
    # print(degree)
    if abs(avg_degree) <= 0.055: # straight
        return 0,avg_degree
    elif avg_degree < 0: # left
        return -1,avg_degree
    else: # right
        return 1,avg_degree

'''
7. 显示结果 (Display results)
'''
def add_icon(img,icon):
    h, w = img.shape[:2]
    img0 = np.zeros_like(img)
    icon = cv2.resize(icon, (160, 80), interpolation=cv2.INTER_AREA)
    cv2.imshow('test1',icon)
    nonzero=icon.nonzero()
    valid_y = np.array(nonzero[0] + h*0.65, dtype="int")
    valid_x = np.array(nonzero[1] + w*0.4, dtype="int")
    img0[valid_y, valid_x] = [255, 255, 255]
    dst = cv2.addWeighted(img, 1, img0, 0.7, 0)
    # cv2.imshow('test1',dst)
    return dst

# 图像合并
def merge_img(img_add, img_per, img_filter):
    # 合并后的车道识别图像，透视变换彩图，透视变换二值图
    # 图像分为左侧(1200，750)车道区域与信息，右侧2部分小图（400，350），分别显示剩余2图
    final_img = np.zeros((750, 1600, 3), dtype=np.uint8)
    final_img.fill(255)
    final_img[0:750, 0:1200, :] = cv2.resize(img_add, (1200, 750))

    # 右侧图片
    final_img[0:350, 1200:1600, :] = cv2.resize(img_per, (400, 350))
    # 图片，内容，位置，字体，大小，颜色和粗细
    cv2.putText(final_img, 'Perspective img', (1300, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    gray_image = cv2.cvtColor(img_filter, cv2.COLOR_GRAY2RGB)
    final_img[400:750, 1200:1600, :] = cv2.resize(gray_image, (400, 350))
    cv2.putText(final_img, 'Filter & ROI ', (1300, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)
    return final_img

def draw_lines(gray_img, color_img, line1, line2):
    # 彩图color img，左线left line，右线right line
    color = np.copy(color_img)
    # 左低 left-line low point，右高 high point
    list1 = np.append(line1[0], line1[1])
    # 右低 right-line low point，左高 high point
    list2 = np.append(line2[0], line2[1])
    # 计算最低点 calculate the lowest point
    p1 = calculate_x([list1])
    p2 = calculate_x([list2])
    cv2.fillPoly(color_img, np.array([[[line1[1][0], line1[1][1]], p1,
                                       p2, [line2[0][0], line2[0][1]]]]), (100, 87, 249))
    cv2.line(color_img, tuple(line1[1]), tuple(p1), color=(0, 255, 0), thickness=2)
    cv2.line(color_img, tuple(p2), tuple(line2[0]), color=(255, 0, 0), thickness=2)

    # 灰度图划线
    img_l = np.zeros_like(gray_img)
    img_r = np.zeros_like(gray_img)
    cv2.line(img_l, tuple(line1[1]), tuple(p1), color=255, thickness=2)
    cv2.line(img_r, tuple(p2), tuple(line2[0]), color=255, thickness=2)
    warped_grayl = perspective_img(img_l)
    warped_grayr = perspective_img(img_r)
    warped_color = perspective_img(color)
    state, mean_degree = direction(warped_grayl, warped_grayr)
    # cv2.imshow('0', warped_grayl)
    # cv2.imshow('1', warped_grayr)
    # cv2.imshow('2', warped_color)

    # 计算斜率 calculate slopes
    l_s=calculate_slope([list1])
    r_s = calculate_slope([list2])
    l_d = abs(np.degrees(np.arctan(l_s)))
    r_d = abs(np.degrees(np.arctan(r_s)))
    cv2.putText(color_img, '{0:6.4f} , {1:6.4f}'.format(l_d,r_d), (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(color_img, '{0:6.4f}'.format(mean_degree), (0, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    if state==0:
        # cv2.putText(color_img, 'go straight', (0, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 5)
        color_img = add_icon(color_img, icon0)
    elif state==-1:
        # cv2.putText(color_img, 'turn left', (0, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 5)
        color_img = add_icon(color_img, icon1)
    else:
        # cv2.putText(color_img, 'turn right', (0, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 5)
        color_img = add_icon(color_img, icon2)
    '''
    l_h, r_h = frame_img.update(line1[1][1], line2[0][1])
    # 高度判断是否转向 Judging whether to turn according to height
    if l_h < r_h and abs(r_h - l_h) > 5:
        cv2.putText(color_img, 'turn right', (0, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 5)
    elif l_h > r_h and abs(r_h - l_h) > 5:
        cv2.putText(color_img, 'turn left', (0, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 5)
    else:
        cv2.putText(color_img, 'go straight', (0, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 5)
    '''
    final=merge_img(color_img, warped_color, gray_img)
    return final

# Previous function integration
# 之前函数整合实现车道线识别
def show_lane(color_img):
    """
    在 color_img 上画出车道线  (draw the lane on the color_img)
    :param color_img：彩色图，channels=3
    :return:
    """
    edge_img, img0 = get_edge_img(color_img)
    mask_gray_img = roi_mask(edge_img)
    llines, rlines = get_lines(mask_gray_img)
    final = draw_lines(mask_gray_img, img0, llines, rlines)
    return final
    # return mask_gray_img

# to get the average data
class Frame:
    def __init__(self, average=5):
        # fixed length queue- 15
        self.lane_l = deque(maxlen=average * 3)     # left line
        self.lane_r = deque(maxlen=average * 3)     # right line
        self.degree = deque(maxlen=average)         # Deviation

    def update(self, degree):
        self.degree.append(degree)
        return np.mean(self.degree)

class VideoStreamingTest(object):
    def __init__(self, host, port):

        self.server_socket = socket.socket()
        self.server_socket.bind((host, port))
        self.server_socket.listen(0)
        self.connection, self.client_address = self.server_socket.accept()
        self.connection = self.connection.makefile('rb')
        self.host_name = socket.gethostname()
        self.host_ip = socket.gethostbyname(self.host_name)
        self.streaming()

    def streaming(self):

        try:
            print("Host: ", self.host_name + ' ' + self.host_ip)
            print("Connection from: ", self.client_address)
            print("Streaming...")
            print("Press 'q' to exit")

            # need bytes here
            stream_bytes = b' '
            while True:
                stream_bytes += self.connection.read(1024)
                first = stream_bytes.find(b'\xff\xd8')
                last = stream_bytes.find(b'\xff\xd9')
                if first != -1 and last != -1:
                    jpg = stream_bytes[first:last + 2]
                    stream_bytes = stream_bytes[last + 2:]
                    image = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    cv2.imshow('image', image)
                    rImg = show_lane(image)
                    cv2.imshow('result', rImg)
                    '''
                    msg='$00#00$' # 前进
                    self.connection.send(msg.encode())
                    '''
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        finally:
            self.connection.close()
            self.server_socket.close()


if __name__ == '__main__':
    frame_img = Frame()
    icon0 = cv2.imread('arrow0.png')
    icon1 = cv2.imread('arrow1.png')
    icon2 = cv2.imread('arrow2.png')
    # host, port
    h, p = "192.168.0.4", 8000
    VideoStreamingTest(h, p)