#!/usr/bin/env python
# coding: utf-8

# # KNN

# In[2]:


import numpy as np
import cv2
cap = cv2.VideoCapture('klsq20201214115453.mp4')
bs = cv2.createBackgroundSubtractorKNN(detectShadows=True)
while(1):
    ret, frame = cap.read()
    fgmask = bs.apply(frame)
    cv2.imshow('frame',fgmask)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()


# # GMG背景差法
# 

# In[3]:


import numpy as np
import cv2
cap = cv2.VideoCapture('klsq20201214115453.mp4')
fgbg = cv2.bgsegm.createBackgroundSubtractorGMG()
while(1):
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    cv2.imshow('frame',fgmask)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()


# # MOG背景差法

# In[10]:


import numpy as np
import cv2
cap = cv2.VideoCapture('klsq20201214115453.mp4')
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()
while(1):
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    cv2.imshow('frame',fgmask)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()


# # MOG2背景差法

# In[1]:


import numpy as np
import cv2
cap = cv2.VideoCapture('klsq20201214115453.mp4')
fgbg = cv2.createBackgroundSubtractorMOG2()
while(1):
    ret, frame = cap.read()
    fgmask = fgbg.apply(frame)
    cv2.imshow('frame',fgmask)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()


# # 中间帧参照法

# ## 自适应阈值处理

# In[2]:


import numpy as np
import cv2
# 读取视频文件
cap = cv2.VideoCapture('zlhy20201214115454.mp4')
# 随机取25帧的index
frameIds = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=25)
# 取出的图片放入数组中
frames = []
for fid in frameIds:
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    ret, frame = cap.read()
    frames.append(frame)
#按照时间顺序计算中值画面
medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)    
grayMedianFrame = cv2.cvtColor(medianFrame, cv2.COLOR_BGR2GRAY)

#展示中间帧画面
cv2.imshow('gray1', medianFrame)
cv2.waitKey(1)

fgbg = cv2.createBackgroundSubtractorMOG2(history=1000, varThreshold=1000, detectShadows=False)


maxval = 255
adaptmethod = 0
thresholdtype = 1
blocksize = 3
constval = 10
while(cap.isOpened()):
  # 读取视频
    ret, frame = cap.read()
  # 转换为灰度图
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  # 计算当前帧和中位帧的不同处
    dframe = cv2.absdiff(frame, grayMedianFrame)
  # 过滤低灰度区域
    dframe = cv2.adaptiveThreshold(dframe,maxval,adaptmethod,thresholdtype,blocksize,constval)
  # 展示
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.imshow('frame', dframe)
    if cv2.waitKey(1) & 0xff == 27:
        break
cap.release()
cv2.destroyAllWindows()


# ## 读取中间背景图像

# In[ ]:


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
#按照时间顺序计算中值画面
medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)    
#展示画面
cv2.imshow('frame', medianFrame)
cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()


# ## 阈值方法处理

# In[ ]:


import numpy as np
import cv2
# 读取视频
cap = cv2.VideoCapture('klsq20201214115453.mp4')
# 随机选取n帧视频
frameIds = cap.get(cv2.CAP_PROP_FRAME_COUNT) * np.random.uniform(size=25)
# 存储对应帧数
frames = []
counter=0
for fid in frameIds:
    counter+=1
    print(counter)
    cap.set(cv2.CAP_PROP_POS_FRAMES, fid)
    ret, frame = cap.read()
    frames.append(frame)
#按照时间顺序计算中值画面
medianFrame = np.median(frames, axis=0).astype(dtype=np.uint8)    
#转换为灰度图
grayMedianFrame = cv2.cvtColor(medianFrame, cv2.COLOR_BGR2GRAY)
cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
ret = True
while(ret):
  # 读取视频
    ret, frame = cap.read()
  # 转换为灰度图
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  # 计算当前帧和中位帧的不同处
    dframe = cv2.absdiff(frame, grayMedianFrame)
  # 过滤低灰度区域
    th, dframe = cv2.threshold(dframe, 30, 255, cv2.THRESH_BINARY)
  # 展示
    cv2.imshow('frame', dframe)
    if cv2.waitKey(1) & 0xff == 27:
        break
cap.release()
cv2.destroyAllWindows()

