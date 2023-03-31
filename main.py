#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: qjk

import cv2
import numpy as np

# video_file = "video_1.mp4"
# video = cv2.VideoCapture(video_file)
camera = cv2.VideoCapture(0)  # 0表示使用第一个摄像头

while True:
    # (grabbed, frame) = video.read()
    # if not grabbed:
    #     break
    ret, frame = camera.read()
    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
    img = frame
    redThre = 115  # 115~135红色分量阈值
    sThre = 60  # 55~65饱和度阈值

    B = img[:, :, 0]
    G = img[:, :, 1]
    R = img[:, :, 2]

    B1 = img[:, :, 0] / 255
    G1 = img[:, :, 1] / 255
    R1 = img[:, :, 2] / 255
    minValue = np.array(
        np.where(R1 <= G1, np.where(G1 <= B1, R1, np.where(R1 <= B1, R1, B1)), np.where(G1 <= B1, G1, B1)))
    sumValue = R1 + G1 + B1
    # HSI中S分量计算公式
    S = np.array(np.where(sumValue != 0, (1 - 3.0 * minValue / sumValue), 0))
    Sdet = (255 - R) / 20
    SThre = ((255 - R) * sThre / redThre)
    # 判断条件
    fireImg = np.array(
        np.where(R > redThre, np.where(R >= G, np.where(G >= B, np.where(S > 0, np.where(S > Sdet, np.where(
            S >= SThre, 255, 0), 0), 0), 0), 0), 0))

    gray_fireImg = np.zeros([fireImg.shape[0], fireImg.shape[1], 1], np.uint8)
    gray_fireImg[:, :, 0] = fireImg
    meBImg = cv2.medianBlur(gray_fireImg, 5)
    kernel = np.ones((5, 5), np.uint8)
    ProcImg = cv2.dilate(meBImg, kernel)
    # 绘制矩形框
    contours, _ = cv2.findContours(ProcImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ResImg = img.copy()
    for c in range(0, len(contours)):
        # 获取矩形的左上角坐标(x,y)，以及矩形的宽和高w、h
        x, y, w, h = cv2.boundingRect(contours[c])
        l_top = (x, y)
        r_bottom = (x + w, y + h)
        cv2.rectangle(ResImg, l_top, r_bottom, (255, 0, 0), 2)
    cv2.imshow("RESULT", ResImg)
    # blur = cv2.GaussianBlur(frame, (21, 21), 0)
    # hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    #
    # lower = [30, 50, 50]
    # upper = [60, 255, 255]
    # lower = np.array(lower, dtype="uint8")
    # upper = np.array(upper, dtype="uint8")
    # mask = cv2.inRange(hsv, lower, upper)


    # output = cv2.bitwise_and(frame, hsv, mask=mask)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
camera.release()
