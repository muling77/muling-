import cv2
import numpy as np
import matplotlib.pyplot as plt
#import math
#import os


#讀取模型與訓練權重
def initNet():
    CONFIG = 'yolov4-custom.cfg'
    WEIGHT = 'yolov4-nine_final.weights'

    net   = cv2.dnn.readNet(CONFIG,WEIGHT)
    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(size=(832,480),scale=1/255.0)
    model.setInputSwapRB(True)

    return model

#物件偵測
def nnProcess(image, model):
    classes, confs, boxes = model.detect(image, 0.4, 0.1)
    return classes, confs, boxes

def getstart(classes, confs, boxes):
    aa,bb = 0, 0
    for (classid, conf, box) in zip(classes, confs, boxes):
        aa=classid
        bb=conf
    return aa,bb

def linearDraw (img,id,xs,ys,ws,hs,p1x,p1y,p2x,p2y,p3x,p3y,p4x,p4y,p5x,p5y,p6x,p6y,):
    pts = np.array([[p1x,p1y], [p2x,p2y], [p3x,p3y], [p4x,p4y], [p5x,p5y], [p6x,p6y]])  
    pts_fit2 = np.polyfit(pts[:, 0], pts[:, 1], 2)
    pts_fit3 = np.polyfit(pts[:, 0], pts[:, 1], 3)
        
    plotx = np.linspace(p1x, p6x, 400)
    ploty2 = pts_fit2[0]*plotx**2 + pts_fit2[1]*plotx + pts_fit2[2]
    ploty3 = pts_fit3[0]*plotx**3 + pts_fit3[1]*plotx**2 + pts_fit3[2]*plotx + pts_fit3[3]
    pts_fited2 = np.array([np.transpose(np.vstack([plotx, ploty2]))])
    pts_fited3 = np.array([np.transpose(np.vstack([plotx, ploty3]))]) 
    #cv2.polylines(img, np.int32([pts]), False, (255, 0, 0), 8)
    #cv2.polylines(img, np.int_([pts_fited2]), False, (0, 0, 255), 5)

    if id==0:
        arrow = np.array([[p6x-11,p6y-2],[(p5x+p6x)*0.5,p6y+(0.1*hs)],[(p5x+p6x)*0.55,p6y-(0.1*hs)]])
    if id==2:
        p7x=p6x-0.4*ws
        cv2.line(img,np.int32((p6x,p6y)),np.int32((p7x,p6y)),(245,206,155),8)
        arrow = np.array([[p7x-7,p6y],[(p7x+p6x)*0.47,p6y+(0.1*hs)],[(p7x+p6x)*0.47,p6y-(0.1*hs)]])

    if id==3:
        arrow = np.array([[p6x+10,p6y-3],[(p5x+p6x)*0.5,p6y+(0.1*hs)],[(p5x+p6x)*0.45,p6y-(0.1*hs)]])
    
    cv2.polylines(img, np.int_([pts_fited3]), False, (245,206,155), 10)
    cv2.fillPoly(img,np.int32([arrow]),(245,206,155))
    return img


def draw_arrow(image, classes, confs, boxes):
    new_image = image.copy()
    for (classid, conf, box) in zip(classes, confs, boxes):
        x, y, w, h = box

        if classid == 0:
            #img=cv2.rectangle(new_image, (x , y ), (x + w , y + h ), (0, 255, 0), 3) #r2 green
            img=linearDraw (new_image,classid,x,y,w,h,x+0.6*w,y+h,x+0.6*w,y+0.8*h,x+0.475*w,y+0.73*h,x+0.35*w,y+0.69*h,x+0.28*w,y+0.66*h,x+0.1*w,y+0.68*h)
            
        if classid == 2:
            #img=cv2.rectangle(new_image, (x , y ), (x + w , y + h ), (0, 0, 255), 3) #r3 red
            img=linearDraw (new_image,classid,x,y,w,h,x+0.8*w,y+h,x+0.8*w,y+0.85*h,x+0.85*w,y+0.74*h,x+0.9*w,y+0.7*h,x+0.95*w,y+0.68*h,x+w,y+0.68*h)
        if classid == 3:
            #img=cv2.rectangle(new_image, (x , y ), (x + w , y + h ), (0, 255, 255), 3) #r4 yellow
            img=linearDraw (new_image,classid,x,y,w,h,x+0.6*w,y+h,x+0.6*w,y+0.71*h,x+0.7*w,y+0.69*h,x+0.8*w,y+0.65*h,x+0.9*w,y+0.62*h,x+w,y+0.62*h)

    return img

if __name__ == '__main__':
    c,t,lastclass=0,0,0
    ret, frame = 0,0

    cap = cv2.VideoCapture('test_r3_copy.mp4')
    model = initNet()
    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    while True:
        ret, frame = cap.read()  #逐幀取圖
        if not ret:
            print("Cannot receive frame")
            break
        if (c % 8==0):  #每8幀取1張
            height, width = frame.shape[:2]
            center = (width // 2, height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, 15, scale=1.0)  #旋轉15度
            img = cv2.warpAffine(frame, rotation_matrix, (width, height))
            #img= cv2.resize(ratate_img, (1920, 1080), interpolation=cv2.INTER_CUBIC)

            classes, confs, boxes = nnProcess(img, model)
            classidd, conff = getstart(classes, confs, boxes)
            if conff > 0.90 and classidd==lastclass :
                t=t+1
            elif 0.85<conff < 0.90 and classidd==lastclass :
                t=t-8
                if t<0:
                    t=0
            else :
                t=0
            lastclass=classidd
            print(lastclass,conff,t)          ###FIX###USE FOR DEBUG!!!!!!!!!!
            if t>7 :
                draw_img=draw_arrow(img, classes, confs, boxes)
                cv2.namedWindow('a',0)
                cv2.resizeWindow('a',832, 480)
                cv2.imshow('a',draw_img)
                cv2.waitKey(1)
            else :
                cv2.namedWindow('a',0)
                cv2.resizeWindow('a',832, 480)
                cv2.imshow('a',img)
                cv2.waitKey(1)
        c=c+1
    cap.release()
    cv2.destroyAllWindows()