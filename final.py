import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np

def Gaussian(gray):
    '''  高斯滤波降噪

    :param gray: 灰度化后的图像
    :return: 降噪后的图像
    '''
    dst = cv.GaussianBlur(gray, (7,7), sigmaX=2.3)

    return dst

def Median(gray):
    '''  中值滤波降噪

    :param gray: 灰度化后的图像
    :return: 降噪后的图像
    '''
    dst = cv.medianBlur(gray, 7)

    return dst

def Sobel(gray):
    '''  Sobel滤波器

    :param gray: 灰度化后的图像
    :return: x, y方向边缘强度
    '''
    sobelx = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=3)
    sobely = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=3)

    plt.subplot(1, 2, 1), plt.imshow(sobelx, cmap='gray')
    plt.title('SobelX'), plt.xticks([]), plt.yticks([])
    plt.subplot(1, 2, 2), plt.imshow(sobely, cmap='gray')
    plt.title('SobelY'), plt.xticks([]), plt.yticks([])
    plt.show()

    return sobelx, sobely

def Edge(fx, fy):
    '''  求边缘梯度，量化梯度方向

    :param fx: x方向边缘强度
    :param fy: y方向边缘强度
    :return: 边缘梯度edge，梯度方向_angle
    '''
    fx_float = fx.astype(np.float32)
    fy_float = fy.astype(np.float32)

    # 代入公式
    edge = pow((pow(fx_float, 2) + pow(fy_float, 2)), 0.5)
    edge = np.clip(edge, 0, 255)

    fx = np.maximum(fx, 1e-10)
    fy = np.maximum(fy, 1e-10)
    tan = np.arctan(fy / fx)  # 边缘梯度

    # 量化梯度方向
    angle = tan / np.pi * 180
    angle[angle < -22.5] = 180 + angle[angle < -22.5]
    _angle = np.zeros_like(angle, dtype = np.uint8)
    _angle[np.where(angle <= 22.5)] = 0
    _angle[np.where((angle > 22.5) & (angle <= 67.5))] = 45
    _angle[np.where((angle > 67.5) & (angle <= 112.5))] = 90
    _angle[np.where((angle > 112.5) & (angle <= 157.5))] = 135

    return edge, _angle

def NonMax(edge, angle):
    '''  非最大化抑制

    :param edge: 边缘梯度
    :param angle: 梯度方向
    :return: 非最大化抑制处理后的边缘梯度
    '''
    H, W = angle.shape
    _edge = edge.copy()

    for y in range(H):
        for x in range(W):
            # 不同方向的dx, dy
            if angle[y, x] == 0:
                dx1, dy1, dx2, dy2 = -1, 0, 1, 0
            elif angle[y, x] == 45:
                dx1, dy1, dx2, dy2 = -1, 1, 1, -1
            elif angle[y, x] == 90:
                dx1, dy1, dx2, dy2 = 0, -1, 0, 1
            elif angle[y, x] == 135:
                dx1, dy1, dx2, dy2 = -1, -1, 1, 1

            # 考虑(x, y)是边缘的情况
            if x == 0:
                dx1 = max(dx1, 0)
                dx2 = max(dx2, 0)
            if y == 0:
                dy1 = max(dy1, 0)
                dy2 = max(dy2, 0)
            if x == W - 1:
                dx1 = min(dx1, 0)
                dx2 = min(dx2, 0)
            if y == H - 1:
                dy1 = min(dy1, 0)
                dy2 = min(dy2, 0)

            # 若非极大，则抑制
            if edge[y, x] < max(edge[y + dy1, x + dx1], edge[y + dy2, x + dx2]):
                _edge[y, x] = 0

    return _edge

def Thresh(edge, low, high):
    '''  双阈值筛选

    :param edge: 边缘梯度
    :param low: 低阈值
    :param high: 高阈值
    :return: 阈值处理后的边缘
    '''
    H, W = edge.shape

    # 周围再加一圈，简化之后的代码
    _edge = np.zeros((H + 2, W + 2), dtype=np.float32)
    _edge[1:H+1, 1:W+1] = edge

    t = np.array(((1, 1, 1), (1, 0, 1), (1, 1, 1)), dtype=np.float32)

    for y in range(1, H + 2):
        for x in range(1, W + 2):
            # 大于高阈值设为255， 小于低阈值设为0
            if _edge[y, x] < low:
                _edge[y, x] = 0
            elif _edge[y, x] > high:
                _edge[y, x] = 255
            else:
                # 周围一圈如果有大于高阈值的，则将该点设为255，反之设为0
                if np.max(_edge[y-1:y+2, x-1:x+2] * t) >= high:
                    _edge[y, x] = 255
                else:
                    _edge[y, x] = 0

    edge = _edge[1:H+1, 1:W+1]  # 删掉加上的那圈

    return edge

def Hough(img, edges):
    '''  霍夫变换

    :param img: 待处理的图像
    :param edges: 检测到的边
    :return: 拟合结果
    '''
    lines = cv.HoughLines(edges.astype('uint8'), 1, np.pi / 180, 240)

    # 绘画拟合结果
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        cv.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return img

def main():
    img_bgr = cv.imread('milk.jpg')
    # h, w = img.shape[:2]
    img = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

    gaussian = Gaussian(gray)
    median = Median(gray)
    sobelx, sobely = Sobel(gray)
    edge, angle = Edge(sobelx, sobely)
    _edge = NonMax(edge, angle)
    edge_thresh = Thresh(_edge, 175, 200)  # 低阈值175， 高阈值200
    img_hough = img.copy()  # 不用copy会覆盖原本的img
    img_hough = Hough(img_hough, edge_thresh)

    # 六张图放在一起做个对比
    plt.subplot(2, 3, 1), plt.imshow(img)
    plt.title('origin'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 3, 2), plt.imshow(gray, cmap="gray")
    plt.title('gray'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 3, 3), plt.imshow(gaussian, cmap="gray")
    plt.title('gaussian'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 3, 4), plt.imshow(median, cmap="gray")
    plt.title('median'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 3, 5), plt.imshow(edge_thresh, cmap="gray")
    plt.title('thresh'), plt.xticks([]), plt.yticks([])
    plt.subplot(2, 3, 6), plt.imshow(img_hough)
    plt.title('hough'), plt.xticks([]), plt.yticks([])
    plt.show()


    '''
    用cv自带的imshow时，
    thresh的结果显示受计算机窗口大小影响
    （在我的电脑上窗口比例不同，结果看起来是不一样的，可能是显示屏分辨率的问题），
    所以为了准确性选择了保存到本地的方式。
    '''
    # cv.namedWindow("orgin", 0) # 可伸缩
    # cv.imshow("orgin", img_bgr)
    # cv.namedWindow("gray", 0)
    # cv.imshow("gray", gray)
    # cv.namedWindow("gaussian", 0)
    # cv.imshow("gaussian", gaussian)
    # cv.namedWindow("thresh", 0)
    # cv.imshow("thresh", edge_thresh)
    # cv.namedWindow("hough", 0)
    # cv.imshow("hough", hough)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    cv.imwrite('gray.jpg', gray)
    cv.imwrite('median.jpg', median)
    cv.imwrite('gaussian.jpg', gaussian)
    cv.imwrite('edge_thresh.jpg', edge_thresh)
    hough = cv.cvtColor(img_hough, cv.COLOR_RGB2BGR)
    cv.imwrite('hough.jpg', hough)

if __name__ == '__main__':
    main()