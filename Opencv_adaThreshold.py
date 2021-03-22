# 自适应阈值方法实现
import numpy as np
import cv2
# 读取视频文件
cap = cv2.VideoCapture('klsq20201214115453.mp4')
# 随机取25帧的index
frameIds = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=25)
# 取出的图片放入数组中
frames = []
for fid in frameIds:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    ret, frame = cap.read()
    frames.append(frame)
# 按照时间顺序计算中值画面
medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)
# 将中值画面转化为灰度图像
grayMedianFrame = cv2.cvtColor(medianFrame, cv2.COLOR_BGR2GRAY)
'''
#展示中间帧画面
cv2.imshow('frame', medianFrame)
cv2.waitKey(1)
'''
grayedges = cv2.Canny(grayMedianFrame,50,200)
cv2.imshow('frame', grayedges)
cv2.waitKey(0)

fgbg = cv2.createBackgroundSubtractorMOG2()

cap.set(0,0)
maxval = 255 # 预设满足条件的最大值
adaptmethod = 0 # 指定自适应阈值算法
thresholdtype = 1 # 指定阈值类型
blocksize = 11 # 表示邻域块大小
constval = 10 # 表示与算法有关的参数
kernel = np.ones((5,5),np.float32)/25
while(cap.isOpened()):
    # 读取视频
    ret, frame = cap.read()
    # 转换为灰度图
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 计算当前帧和中位帧的不同处
    dframe = cv2.absdiff(frame, grayMedianFrame)
    # 过滤低灰度区域
    dframe = cv2.adaptiveThreshold(dframe,maxval,adaptmethod,thresholdtype,blocksize,constval)
    # 背景去除算法
    fgframe = fgbg.apply(dframe)
    # 2D滤波器
    dst = cv2.filter2D(fgframe,-1,kernel)
    # 高斯滤波器降噪
    blur = cv2.GaussianBlur(dst,(5,5),0)
    # Canny边缘检测
    edges = cv2.Canny(blur,100,200)
    # 展示
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.imshow('frame', edges)
    if cv2.waitKey(1) & 0xff == 27:
        break
cap.release()
cv2.destroyAllWindows()