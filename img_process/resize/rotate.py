import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def rotate_arrow(img,i):

# 指定旋轉角度
    angle = 15  # 以度為單位的角度
# 計算圖像中心
    height, width = img.shape[:2]
    center = (width // 2, height // 2)

# 創建旋轉矩陣
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale=1.0)

# 執行旋轉
    rotated_image = cv2.warpAffine(img, rotation_matrix, (width, height))

# 顯示旋轉後的圖像
    cv2.imshow('img%d'%(i), rotated_image)
    cv2.imwrite('E:\\fcu_data\\car\\road3_set1\\%d.jpg'%i, rotated_image)
    cv2.waitKey(2)
   



if __name__ == '__main__':
    for i in range(1,103):                                                      
        img=cv2.imread("E:\\fcu_data\\car\\road3_set\\%d.jpg"%(i))
       
        rotate_arrow(img,i)
      



