from segmentation import lineSegmentation 
from preprocessing import imgPreProcess
import pickle
import numpy as np

fileName = 'example3.png'

sizeOfImage = 64
Actual_Y = {0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9', 10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H', 18: 'I', 19: 'J', 20: 'K', 21: 'L', 22: 'M', 23: 'N', 24: 'O', 25: 'P', 26: 'Q', 27: 'R', 28: 'S', 29: 'T', 30: 'U', 31: 'V', 32: 'W', 33: 'X', 34: 'Y', 35: 'Z', 36: 'a', 37: 'b', 38: 'c', 39: 'd', 40: 'e', 41: 'f', 42: 'g', 43: 'h', 44: 'i', 45: 'j', 46: 'k', 47: 'l', 48: 'm', 49: 'n', 50: 'o', 51: 'p', 52: 'q', 53: 'r', 54: 's', 55: 't', 56: 'u', 57: 'v', 58: 'w', 59: 'x', 60: 'y', 61: 'z'}
Y = ['This','is','handwritten','example','Write','as','good','as','you','can']
count =0
finalOp = ""
data = lineSegmentation(fileName)

for word in data:
    X = imgPreProcess(word,sizeOfImage)
    X = (np.repeat(np.array(X)[...,np.newaxis],3,-1))
    # print(X)
    with open('vggNetModelCnn.sav','rb') as f:
        model = pickle.load(f)
    predict_x=model.predict(X) 
    y_pred=np.argmax(predict_x,axis=1)
    output = ""
    for i in y_pred:
        # i = i.tolist()
        # result = i.index(max(i))
        # print(result)
        output += Actual_Y[i]
    finalOp += output + " "
    
print(finalOp)
    # count+=1
    # print('Test loss:', score[0])
    # print('Test accuracy:', score[1])
# c = Console.getconsole()
# c.text(0, -1, 'And this is the string at the bottom of the console')
