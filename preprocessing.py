import os
import cv2
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import numpy as np
from pathlib import Path
from xlwt import Workbook
from  xlrd import open_workbook
# import openpyxl


from PIL import Image


def toRGB(img):
    converted = tf.image.grayscale_to_rgb(img)
    return converted


def preTempProcess(size_):

    folder_dir = "..\AugmentedDataset"
    # paths = sorted(Path(folder_dir).iterdir(), key=os.path.getctime)
    image_array = []
    normalized_array = []
    scalar = MinMaxScaler()
    counter = 0
    addition = 200
    image_array = []
    # print(os.listdir(folder_dir))

    files = []

    for i in os.listdir(folder_dir):
        files.append(folder_dir+'/'+i)

    files.sort(key=os.path.getctime)
    # print(files)

    for images in files:
        # if(counter % addition != 0):
        #     counter += 1
        #     continue

        if (images.endswith(".png")):
            counter += 1
            img = cv2.imread(images, 0)

            img_resized = cv2.resize(img, (size_, size_))
            image_array.append(img_resized)
            plt.imshow(img_resized, cmap='gray')
            plt.show()
            # print(img_resized)
            last_axis = -1
            dim_to_repeat = 2
            repeats = 3
            grscale_img_3dims = np.expand_dims(img_resized, last_axis)
            img_rgb = toRGB(tf.constant(grscale_img_3dims))
            image_array.append(img_rgb)
            scalar.fit(img_resized)
            normalized_array.append(scalar.transform(img_resized))
    # print(normalized_array[0])
    return (normalized_array, image_array)


def preProcess(size_):

    # folder_dir = "..\Dataset\Img"
    folder_dir = "..\AugmentedDataset"


    image_array = []
    normalized_array = []
    scalar = MinMaxScaler()
    files = []

    for i in os.listdir(folder_dir):
        files.append(folder_dir+'/'+i)

    files.sort(key=os.path.getctime)
    # print(files)

    for images in files:
        if (images.endswith(".png")):
            img = cv2.imread(images, 0)
            img_resized = cv2.resize(img, (size_, size_))
            # plt.imshow(img_resized,cmap='gray')
            # plt.show()
            # print(img_resized)
            last_axis = -1
            dim_to_repeat = 2
            repeats = 3
            grscale_img_3dims = np.expand_dims(img_resized, last_axis)
            img_rgb = toRGB(tf.constant(grscale_img_3dims))
            image_array.append(img_rgb)
            scalar.fit(img_resized)
            normalized_array.append(scalar.transform(img_resized))
    return normalized_array


def imgPreProcess(imgArray, size_):
    normalized_array = []
    scalar = MinMaxScaler()
    # kernalChar = np.ones((5, 5), np.uint8)

    for img in imgArray:

        dimension = 120

        height, width = img.shape
        x_pad = 0
        y_pad = 0

        if (dimension > width):
            x_pad = int((dimension - width)/2)

        if (dimension > height):
            y_pad = int((dimension - height)/2)

        img = cv2.copyMakeBorder(
            img, y_pad, y_pad, x_pad, x_pad, cv2.BORDER_CONSTANT, None, value=255)

        kernalCharLast = np.array([[0, 0, 0],
                                   [0, 0, 0],
                                   [0, 0, 0]], np.uint8)
        image_dialeted = cv2.dilate(img, kernalCharLast)

        # plt.imshow(image_dialeted,cmap="gray")
        # plt.show()

        img_resized = cv2.resize(image_dialeted, (size_, size_))

        # plt.imshow(img_resized,cmap="gray")
        # plt.show()
        scalar.fit(img_resized)

        normalized_array.append(scalar.transform(img_resized))
    return normalized_array


def createDataFromImage(imgArray, size_ , textArray):
    image_array = []
    normalized_array = []
    scalar = MinMaxScaler()
    kernalChar = np.ones((5, 5), np.uint8)
    countText = 0


    # store in folder

    rowCount = 0
    columnNumber = 0
    wb = open_workbook('augmentedData.xlsx')
    ws = wb.sheet_by_index(0)
    rowCount = ws.nrows
    rowCount += 1

    for img in imgArray:

        dimension = 120

        height, width = img.shape
        x_pad = 0
        y_pad = 0

        if (dimension > width):
            x_pad = int((dimension - width)/2)

        if (dimension > height):
            y_pad = int((dimension - height)/2)

        img = cv2.copyMakeBorder(
            img, y_pad, y_pad, x_pad, x_pad, cv2.BORDER_CONSTANT, None, value=255)

        kernalCharLast = np.array([[0, 0, 0],
                                   [0, 0, 0],
                                   [0, 0, 0]], np.uint8)
        image_dialeted = cv2.dilate(img, kernalCharLast)

        # plt.imshow(img,cmap="gray")
        # plt.show()

        img_resized = cv2.resize(image_dialeted, (size_, size_))

        # plt.imshow(image_dialeted,cmap="gray")
        # plt.show()
        # last_axis = -1
        # dim_to_repeat = 2
        # repeats = 3
        # grscale_img_3dims = np.expand_dims(img_resized, last_axis)
        # gray_image = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
        # print(gray_image.shape)
        # img_dialeted= cv2.dilate(gray_image,kernalChar,iterations=1)

        # image_array.append(img_resized)
        scalar.fit(img_resized)

        # store in folder

        # ws.cell(row=rowCount, column=columnNumber).value = 'image' + str(rowCount)+'.png'
        # ws.cell(row=rowCount, column=columnNumber+1).value = 

        # wb.save('C:\\Temp\\exp\\data.xlsx')
        # ws.write(count_modified, 0, 'image' + str(count_modified)+'.png')
        # ws.write(count_modified, 1, df['label'][count_original])

        normalized_array.append(scalar.transform(img_resized))
        return normalized_array