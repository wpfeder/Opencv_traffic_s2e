#帧差法
import numpy as np
import cv2

# 读取视频文件
cap = cv2.VideoCapture('klsq20201214115453.mp4')
fraspd = cap.get(5) # 帧速率提取

# 读取背景图案
# 随机取25帧的index
frameIds = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=25)
# 取出的图片放入数组中
frames = []
for fid in frameIds:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    ret, frame = cap.read()
    frames.append(frame)
#按照时间顺序计算中值画面并将其转化为灰度图像
medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)    
grayMedianFrame = cv2.cvtColor(medianFrame, cv2.COLOR_BGR2GRAY)

# 回调视频位置和初始化
cap.set(1, 0)

tmp1 = None
cv2.namedWindow('frame', cv2.WINDOW_NORMAL)

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret is not True: 
        break
    tmp = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # 用tmp进行储存灰度矩阵
    if tmp1 is None:
        tmp1 = tmp
    else:
        # 计算运动的车的差分
        foreFrame = cv2.absdiff(tmp,tmp1)
        # 计算不动的车
        frame1 = cv2.bitwise_xor(grayMedianFrame,tmp1)
        frame2 = cv2.bitwise_xor(grayMedianFrame,tmp)
        frameadd = cv2.bitwise_and(frame1,frame2)
        stableFrame = cv2.absdiff(foreFrame,frameadd)
        # 更新
        tmp1 = tmp
    # 阈值方法和噪音处理
    _, thresh1 = cv2.threshold(foreFrame, 30, 255, cv2.THRESH_BINARY)
    closing1 = cv2.GaussianBlur(thresh1,(5,5),0)
    # 寻找轮廓
    binary1, contours1, hierarchy1 = cv2.findContours(closing1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 去除杂点轮廓
    for c in contours1: 
        # 忽略小轮廓，排除误差 
        if cv2.contourArea(c) < 300: 
            continue 
        # 计算轮廓的边界框，在当前帧中画出该框 
        (x, y, w, h) = cv2.boundingRect(c) 
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    # 不动车计算
    _, thresh2 = cv2.threshold(stableFrame, 30, 255, cv2.THRESH_BINARY)
    closing2 = cv2.GaussianBlur(thresh2,(5,5),0)
    # 寻找轮廓
    binary2, contours2, hierarchy2 = cv2.findContours(closing2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 去除杂点轮廓
    for c in contours2: 
        # 忽略小轮廓，排除误差 
        if cv2.contourArea(c) < 300: 
            continue 
        # 计算轮廓的边界框，在当前帧中画出该框 
        (x, y, w, h) = cv2.boundingRect(c) 
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2) 
    
    cv2.imshow("frame",frame)
    # 退出设置
    if cv2.waitKey(int(fraspd)) & 0xff == 27:
        break
    
cap.release()
cv2.destroyAllWindows()