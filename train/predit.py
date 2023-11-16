import cv2
import numpy as np
import os
import shutil

#讀取模型與訓練權重
def initNet():
    CONFIG = 'yolov4-tiny-myobj.cfg'
    WEIGHT = 'yolov4-tiny-myobj_last.weights'

    net   = cv2.dnn.readNet(CONFIG,WEIGHT)
    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(size=(416,416),scale=1/255.0)
    model.setInputSwapRB(True)

    return model

#物件偵測
def nnProcess(image, model):
    classes, confs, boxes = model.detect(image, 0.4, 0.1)

    return classes, confs, boxes

#框選偵測到的物件，並裁減
def drawBox(image, classes, confs, boxes):
    new_image = image.copy()
    cut_img_list = []
    for (classid, conf, box) in zip(classes, confs,boxes):
        x,y,w,h = box
        if x - 2 < 0:
            x = 2
        if y - 2 < 0:
            y = 2
        cv2.rectangle(new_image, (x - 2, y - 2), (x + w + 2, y + h + 2), (0, 255, 0), 3)
        cut_img = img[y:y + h + 2, x:x + w + 2]
        cut_img_list.append(cut_img)
    return new_image, cut_img_list[0]

#依照偵測到的物件數量進行分類
def copyClassify(file ,input, boxes, file_name, l, m, n):
    box_num = len(boxes)
    if box_num == 0:
        shutil.copy2(input, './02_yolo_classify3/03_no_word/{}'.format(file_name))
        print('※{}成功複製到no_word'.format(file))
    elif box_num == 1:
        shutil.copy2(input, './02_yolo_classify3/01_word/{}'.format(file_name))
        print('※{}成功複製到word'.format(file))
    else:
        shutil.copy2(input, './02_yolo_classify3/02_words/{}'.format(file_name))
        print('※{}成功複製到words'.format(file))
    print('  沒有字：{}張'.format(l))
    print('  1個字：{}張'.format(m))
    print('  2個字以上：{}張'.format(n))

# 儲存已完成前處理之圖檔(中文路徑)
def saveClassify(image, output, p):
    cv2.imencode(ext='.jpg', img=image)[1].tofile(output)
    print('第{}張框字並儲存成功'.format(p))

if __name__ == '__main__':
    source = './01_origin/' #分類文字數量
    # source = './02_yolo_classify3/01_word/' #裁切一個字的照片
    files = os.listdir(source)
    files.sort(key=lambda x:int(x[:-6])) #依照正整數排序
    model = initNet()
    p, l, m, n = 0, 0, 0, 0
    for file in files:
        img = cv2.imdecode(np.fromfile(source+file,dtype=np.uint8),-1)
        classes, confs, boxes = nnProcess(img, model)
        p += 1
        if len(boxes)==0:
            l += 1
        elif len(boxes)==1:
            m += 1
        else:
            n += 1
        try:
            frame, cut = drawBox(img, classes, confs, boxes)
            frame = cv2.resize(frame, (240, 200), interpolation=cv2.INTER_CUBIC) #框選後的照片
            cv2.imshow('img', frame)
            cut2 = cv2.resize(cut, (80, 60), interpolation=cv2.INTER_CUBIC) #裁剪後的照片
            cv2.imshow('cut', cut2)
            copyClassify(file, source + file, boxes, file, l, m, n) #分類文字數量並儲存(one word or above two word)
            # saveClassify(frame, './02_yolo_classify3/select/'+file, p) #儲存框選後的照片
            # saveClassify(cut2, './02_yolo_classify3/cut2/' + file, p) #儲存裁剪後的照片
            cv2.waitKey()
        except:
            copyClassify(file, source + file, boxes, file, l, m, n) #分類文字數量並儲存(no word)
            continue
    print('程式執行完畢')