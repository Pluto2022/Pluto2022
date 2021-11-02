import os

import cv2
import numpy as np
import matplotlib.pyplot as plt

#读取确定好的侧面图的模板顺序图
Model=cv2.imread('Model.png')
#侧面拍摄图模板的标准像素长宽
tH,tW=Model.shape[:2]
#确定起始点，裁剪特征区域，匹配区域1
feature1 = Model[1400:3400, tW-410:tW] # 裁剪坐标为[y0:y1, x0:x1]
#裁剪特征区域，匹配区域2
feature2 = Model[1400:3400, tW-820:tW-410] # 裁剪坐标为[y0:y1, x0:x1]
# cv2.imshow("feature1",feature1)
#保存起始区域1
cv2.imwrite("Feature1-1.png",feature1)
#保存第二起始区域2
# cv2.imshow("feature2",feature2)
cv2.imwrite("Feature2-1.png",feature2)
#对特征区域进行灰度处理
gray1=cv2.cvtColor(feature1,cv2.COLOR_BGR2GRAY)
gray2=cv2.cvtColor(feature2,cv2.COLOR_BGR2GRAY)

#dir，for循环遍历所有图片进行重新统一命名
DATADIR="./Data/"
path=os.path.join(DATADIR)
img_list=os.listdir(path)
ind=0
for i in img_list:
    img_array=cv2.imread(os.path.join(path,i),cv2.IMREAD_COLOR)
    new_array=cv2.resize(img_array,(tW,tH))
    img_name = str(ind) + '.png'
    save_path='.//Side photos//c'+str(ind)+'.png'
    ind=ind+1
    cv2.imwrite(save_path,new_array)

#这里需要遍历文件夹，对文件夹中的所有图像进行特征识别(还未实现）



def main():
    path = ".\\Side photos\\"  # 图像读取地址
    savepath = ".\\res\\"  # 图像保存地址
    filelist = os.listdir(path)  # 打开对应的文件夹
    total_num = len(filelist)  # 得到文件夹中图像的个数

    for i in range(total_num):
        png_name = path + 'c' + str(i) + '.png'   # 拼接图像的读取地址
        # 对图像数据类型进行转换
        img_src = cv2.imread(png_name,cv2.IMREAD_GRAYSCALE)

        # savepng_name = savepath + 'res' + str(i) + '.png'  # 拼接保存图像的地址
        # # png_name.save(savepng_name) #保存图像
        # cv2.imwrite(savepng_name, image)
        # 1.导入图片 与 模板图片
        # img_src = cv2.imread("./Side photos/s9.png", cv2.IMREAD_GRAYSCALE)
        img_temp1 = cv2.imread('Feature1-1.png', cv2.IMREAD_GRAYSCALE)
        img_temp2 = cv2.imread('Feature2-1.png', cv2.IMREAD_GRAYSCALE)
        temp_h1, temp_w1 = img_temp1.shape[:]
        temp_h2, temp_w2 = img_temp2.shape[:]

        # 2.执行模板匹配
        result1 = cv2.matchTemplate(img_src, img_temp1, cv2.TM_SQDIFF)
        result2 = cv2.matchTemplate(img_src, img_temp2, cv2.TM_SQDIFF)
        # 3.计算模板匹配矩形
        min_val1, max_val1, min_loc1, max_loc1 = cv2.minMaxLoc(result1)
        left_top1 = min_loc1
        right_botton1 = min_loc1[0] + temp_w1
        bottom_right1 = (left_top1[0] + temp_w1, left_top1[1] + temp_h1)

        min_val2, max_val2, min_loc2, max_loc2 = cv2.minMaxLoc(result2)
        left_top2 = min_loc2
        right_botton2 = min_loc2[0] + temp_w2
        bottom_right2 = (left_top2[0] + temp_w2, left_top2[1] + temp_h2)
    #if条件，模板匹配SQDIFF，若匹配最小值小于预设值，则是在区域1下进行定位分割拼接
        if min_val1 <=140000000.0:
            print(max_val1, min_val1)
            print(max_loc1, min_loc1)
            cropped11 = img_src[0:tH, 0:right_botton1]
            cropped12 = img_src[0:tH, right_botton1:tW]
            res1 = np.hstack((cropped12, cropped11))
            cv2.imshow('res1', res1)
            savepng_name = savepath + 'res' + str(i) + '.png'  # 保存图像的地址
            cv2.imwrite(savepng_name, res1)
            # cv2.imwrite('./result/res9.png', res1)
    #若图片中没找到对应匹配值下的区域，那么就从定位模板匹配区域2的矩形区域，去找区域1的一部分，与右侧部分拼接成最终标准侧面图 。

        else:
            print(max_val2,min_val2)
            print(max_loc2,min_loc2)
            cropped21=img_src[0:tH,0:temp_w2-(tW-right_botton2)]
            cropped22=img_src[0:tH,temp_w2-(tW-right_botton2):tW]
            print(right_botton2)
            res2=np.hstack((cropped22,cropped21))
            cv2.imshow('res2', res2)
            savepng_name = savepath + 'res' + str(i) + '.png'  # 保存图像的地址
            cv2.imwrite(savepng_name, res2)
            # cv2.imwrite('./result/res9.png',res2)

    #4.显示区域1和区域2的对应在拍摄图上的匹配结果
    cv2.rectangle(img_src, left_top1, bottom_right1, 255, 10)
    plt.figure("显示结果", figsize=(10, 7))
    cv2.rectangle(img_src, left_top2, bottom_right2, 255, 10)
    plt.figure("显示结果", figsize=(10, 7))

    plt.subplot(121)
    plt.imshow(result1, cmap="gray")
    plt.title("match result")
    plt.xticks([])
    plt.yticks([])
    plt.subplot(121)
    plt.imshow(result2, cmap="gray")
    plt.title("match result")
    plt.xticks([])
    plt.yticks([])


    plt.subplot(122)
    plt.imshow(img_src, cmap="gray")
    plt.title("img_src")
    plt.xticks([])
    plt.yticks([])

    plt.show()

    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

