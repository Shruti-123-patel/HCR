import keras
from keras.models import Sequential
from keras.layers import Dense,Flatten
import numpy as np
from keras.applications import VGG16
import matplotlib.pyplot as plt 
import pickle
from sklearn.metrics import confusion_matrix

Actual_Y = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J', 20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T', 30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z', 36: 'a', 37: 'b', 38: 'c', 39: 'd', 40: 'e', 41: 'f', 42: 'g', 43: 'h', 44: 'i', 45: 'j', 46: 'k', 47: 'l', 48: 'm', 49: 'n', 50: 'o', 51: 'p', 52: 'q', 53: 'r', 54: 's', 55: 't', 56: 'u', 57: 'v', 58: 'w', 59: 'x', 60: 'y', 61: 'z'}

def evaluate(x,y,imgArray):
    # print(y)
    x = np.array(x)
    # finalOp = ""
    x = (np.repeat(x[...,np.newaxis],3,-1))
    with open('vggNetModelCnn.sav','rb') as f:
        model = pickle.load(f)

    predict_x=model.predict(x) 
    print(predict_x)

    y_pred = []
    count = 0 

    for i in predict_x :
       plt.imshow(imgArray[count],cmap='gray')
       plt.show()
       y_pred.append(np.argmax(i))
       print(np.argmax(i)," ",y[count]," ",Actual_Y[np.argmax(i)])
       count+=1

    # print(y_pred)

    # output = ""
    # print(len(y_pred),len(x))
    # for i in y_pred:
        # print(i)
        # output += Actual_Y[i]
    # finalOp += output + " "
    
    # confusion_matrix(y, predict_x)
    # print(output)
    # print(y)