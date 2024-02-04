import cv2
import numpy as np
import sys

# 伽玛变换  power函数实现幂函数

if __name__ == "__main__":
    img = cv2.imread("E:\\fcu_data\\car\\road4\\1.jpg")
    # 归1
    Cimg = img / 255
    # 伽玛变换
    gamma = 0.4
    O = np.power(Cimg,gamma)
    #效果
    cv2.imshow('img',img)
    cv2.imshow('O',O)
    cv2.waitKey(0)
    cv2.destroyAllWindows()