from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from keras.utils import img_to_array, load_img
import os,cv2
import pandas as pd
import xlwt
from xlwt import Workbook


datagen = ImageDataGenerator(
        rotation_range = 15,
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = False)
    
Imgsize = 32

folder_dir = "..\Dataset\Img"
image_array = []
normalized_array = []

df = pd.read_csv("..\Dataset\english.csv")

wb = Workbook()

count_modified = 1

count_original = 0 

sheet1 = wb.add_sheet('Sheet 1')

sheet1.write(0,0,'image')
sheet1.write(0,1,'label')


imgFromCurrImg = 3

for images in os.listdir(folder_dir):
    if (images.endswith(".png")):
        # img=cv2.imread(folder_dir+'/'+images,0)
        
        img = load_img(folder_dir+'/'+images)
        img = img_to_array(img)
        img_resized = img.reshape((1, ) + img.shape) 
        # img_resized = cv2.resize(img, (Imgsize,Imgsize))
        i=0
        for batch in datagen.flow(img_resized, batch_size = 1,
                          save_to_dir ='../AugmentedDataset', 
                          save_prefix ='image'+ str(count_modified), save_format ='png'):
            i += 1
            sheet1.write(count_modified, 0, 'image'+ str(count_modified)+'.png')
            sheet1.write(count_modified, 1, df['label'][count_original])
            count_modified+=1
            if i > imgFromCurrImg:
                break
        count_original +=1

wb.save('augmentedData.xls')