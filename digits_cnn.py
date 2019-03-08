# coding:UTF-8
import socket
import threading
import cv2
import numpy as np
import cnn_classificator


def centralization(image_gray):
    dst = np.zeros((280, 280), dtype=np.uint8)
    for i in range(280):
        for j in range(280):
            dst[i][j] = 255
    # 下面一句是不可取的
    # dst.dtype = 'uint8'

    # 先把原图缩小成一个长方形
    image_gray = cv2.resize(image_gray, (80, 200))

    # 再将这个长方形填入28*28的全0矩阵中
    idk1 = 0
    idk2 = 0
    for i in range(40, 240):
        for j in range(100, 180):
            dst[i][j] = image_gray[idk1][idk2]
            if idk2 == 79:
                idk1 += 1
                idk2 = 0
            else:
                idk2 += 1
    # 膨胀三次
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    for i in range(0, 3):
        dst = cv2.erode(dst, kernel)
    # 缩小至28*28
    dst = cv2.resize(dst, (28, 28))
    # cv2.imshow("DST", dst)
    # cv2.imshow("Gray", image_gray)
    # cv2.waitKey(0)

    return dst


def anti_binarization(image_gray):
    (r, c) = image_gray.shape
    for idk1 in range(r):
        for idk2 in range(c):
            if image_gray[idk1][idk2] > 125:
                image_gray[idk1][idk2] = 0
            else:
                image_gray[idk1][idk2] = 255

    return image_gray


def divPic(image_gray):
    (r, c) = image_gray.shape
    print ("r:", r, " c:", c)
    # 对hello.jpg进行分割处理
    res = []
    # 记录每个数字的上下边界
    horizontal = []
    top = bottom = -1
    for idk2 in range(r):
        if 0 in image_gray[idk2, :] and top == -1:
            top = idk2
        if 0 not in image_gray[idk2, :] and top != -1:
            bottom = idk2
            horizontal.append((top, bottom))
            top = bottom = -1
    # 确定每个数字的左右边界
    left = right = -1
    for pair in horizontal:
        for i in range(c):
            if 0 in image_gray[pair[0]: pair[1], i] and left == -1:
                left = i
            if 0 not in image_gray[pair[0]: pair[1], i] and left != -1:
                right = i
                res.append(image_gray[pair[0]: pair[1], left: right])
                left = -1
                right = -1

    # 将分割完毕的每一个数字显示出来
    # for i, element in enumerate(res):
    #     cv2.imshow("Number" + str(i), element)
    #     cv2.waitKey(0)

    return res


def binarization(image_gray):
    # 二值化
    (r, c) = image_gray.shape
    for idk1 in range(r):
        for idk2 in range(c):
            if image_gray[idk1][idk2] > 125:
                image_gray[idk1][idk2] = 255
            else:
                image_gray[idk1][idk2] = 0

    return image_gray


# 对divPic分割出的单个图像，剪去图像上下的留白
def cutting(image_div):
    (r, c) = image_div.shape
    top = 0
    bottom = r
    for i in range(r):
        if 0 in image_div[i]:
            top = i
            break
    for i in range(top, r):
        if 0 not in image_div[i]:
            bottom = i
            break
    image_rediv = image_div[top: bottom, 0: c]
    return image_rediv


def tcplink(sock, addr):
    print("Connection from %s:%s" % addr)
    f = open('hello.jpg', 'wb')
    while True:
        data = sock.recv(1024)
        if data == 'exit' or not data:
            print ("Recognition!")
            # 图片接收结束，读取图片并灰度化、二值化
            image = cv2.imread('hello.jpg')
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image_gray = binarization(image_gray)
            # 这个divNum是分割后的每个数字的集合，是numpy数组的集合
            divNum = divPic(image_gray)
            # 对每个数字进行识别，把结果字符串发送回去
            res = ""
            for element in divNum:
                test_data_set = []
                # 这里剪去留白
                element = cutting(element)
                # 这里处理成28*28
                image_resize = centralization(element)
                # 这里黑白反转
                image_resize = anti_binarization(image_resize)
                # 这里将28*28转换成1*784，作为神经网络的输入
                for i in range(0, 28):
                    for j in range(0, 28):
                        test_data_set.append(image_resize[i][j])
                test_data = np.array(test_data_set)
                print test_data.shape
                # 这里预测
                prediction = cnn_classificator.predict(test_data)
                res += str(prediction)
            print res
            sock.send(res)
            break
        f.write(data)
    sock.close()
    print("Connection from %s:%s closed." % addr)


def main():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(('', 9999))
    s.listen(1)
    print("Waiting for connection......")
    while True:
        sock, addr = s.accept()
        t = threading.Thread(target=tcplink, args=(sock, addr))
        t.start()


if __name__ == '__main__':
    main()
