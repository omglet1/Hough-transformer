import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image,ImageEnhance


def img_processing(img):
    #图像灰度化
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret,binary = cv2.threshold(gray,0,255,cv2.THRESH_OTSU)
    #canny边缘检测
    edges = cv2.Canny(binary,128,200,apertureSize=3)
    return edges

def line_detect(img):
    #用于直线线段检测
    '''
    :param img:检测图像的存储位置
    '''
    img = Image.open(img)

    plt.subplot(2,2,1)
    plt.imshow(img)
    plt.title("Original")
    plt.axis('off')

    img = ImageEnhance.Contrast(img).enhance(3)
    plt.subplot(2, 2, 2)
    plt.imshow(img)
    plt.title("Enhance")
    plt.axis('off')

    img = np.array(img)
    result = img_processing(img)
    plt.subplot(2, 2, 3)
    plt.imshow(result)
    plt.title("Gray+Canny")
    plt.axis('off')

    #hough直线检测
    lines = cv2.HoughLinesP(result,1,1 * np.pi/180,10,minLineLength=20,maxLineGap=50)
    print(lines)
    print("Line Num （线段）:", len(lines))
    #画出检测出的线段
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(img,(x1,y1),(x2,y2),(255,0,0),1)
        pass
    img = Image.fromarray(img,'RGB')
    #img.show()
    plt.subplot(2, 2, 4)
    plt.imshow(img)
    plt.title("Line detection")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    img_root = 'data/002.jpeg'
    line_detect(img_root)
    pass