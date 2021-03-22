# 三帧法
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

# 回调视频位置
cap.set(1, 0)
# 三帧法实践首先前两帧图像做灰度差，然后当前帧图像与前一帧图像做灰度差，最后1和2的结果图像按位做“与”操作，进行阙值判断和得出运动目标。
frameNum = 0 # 定位frame
# 储存过去两帧
tmp1 = None
tmp2 = None
# 形态学运算算子
kernel = np.ones((5,5),np.uint8)
cv2.namedWindow('frame', cv2.WINDOW_NORMAL) # 调整窗口为可拖动
while(cap.isOpened()):
    ret, frame = cap.read() # 读取下一帧
    if ret is not True: break # 如果不能抓取一帧，说明到了视频结尾
    frameNum += 1
    tmp = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # 用tmp进行储存灰度矩阵
    if frameNum == 1:
        tmp1 = tmp # 储存第一帧
    elif frameNum == 2:
        tmp2 = tmp # 储存第二帧
    elif frameNum > 2:
        # 计算三帧差分
        frameD1 = cv2.absdiff(tmp1, tmp2)  # 帧差1
        frameD2 = cv2.absdiff(tmp2, tmp) # 帧差2
        thresh1 = cv2.bitwise_and(frameD1,frameD2) # 与运算
        # 计算得到不动车
        frameA1 = cv2.add(frameD1,grayMedianFrame)
        frameA2 = cv2.add(frameD2,grayMedianFrame)
        t1 = cv2.bitwise_xor(frameA1,tmp)
        t2 = cv2.bitwise_xor(frameA2,tmp)
        thresh2 = cv2.addWeighted(t1,0.5,t2,0.5,0)
        # 对帧进行更新
        tmp1 = tmp2
        tmp2 = tmp
    _ , thresh1 = cv2.threshold(thresh1, 13, 255, cv2.THRESH_BINARY) # 是否加入自动更新光照影响因子 ?
    _ , thresh2 = cv2.threshold(thresh2, 13, 255, cv2.THRESH_BINARY)
    closing1 = cv2.morphologyEx(thresh1, cv2.MORPH_CLOSE, kernel)
    closing2 = cv2.morphologyEx(thresh2, cv2.MORPH_CLOSE, kernel)
    closing1 = cv2.GaussianBlur(thresh1,(5,5),0)
    closing2 = cv2.GaussianBlur(thresh2,(5,5),0)
    # 用Canny边缘检测进行连通 ?
    # edges = cv2.Canny(closing,20,200,5)
    # 捕捉运动车的轮廓
    binary1, contours1, hierarchy1 = cv2.findContours(closing1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 捕捉不动车的轮廓
    binary2, contours2, hierarchy2 = cv2.findContours(closing2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 忽视小轮廓
    #for c in contours:
    #   if cv2.contourArea(c)<1000:
    #        continue
    # 绘制运动的车辆并将其轮廓标绿
    cv2.drawContours(frame, contours1, -1, (0,255,0), 3)
    # 绘制不运动的车辆并将其轮廓标红
    cv2.drawContours(frame, contours2, -1, (0,0,255), 3)    
    cv2.imshow("frame",frame)
    if cv2.waitKey(int(fraspd)) & 0xff == 27:
        break
    
cap.release()
cv2.destroyAllWindows()