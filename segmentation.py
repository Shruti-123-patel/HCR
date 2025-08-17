import cv2
import numpy as np
import matplotlib.pyplot as plt

def lineSegmentation(imgName):
    img = cv2.imread(imgName)
    chars = []
    print(img)
    h,w,c = img.shape
    wi = w
    # if w>1000:
    #     new_w = 1000
    #     ar = (w/h)
    #     new_h = int(new_w/ar)
    #     img = cv2.resize(img,(new_w,new_h),interpolation=cv2.INTER_AREA)

    # Needed because threshold can be done on grayscale image
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # plt.imshow(img,cmap="gray")
    # plt.show()

    dim,imgCopy = cv2.threshold(img,80,255,cv2.THRESH_BINARY)
    # print("Copy")
    # print(imgCopy)

    # plt.imshow(imgCopy,cmap="gray")
    # plt.show()

    

    # because contour want binary img only
    ret,thrsImg = cv2.threshold(img,80,255,cv2.THRESH_BINARY_INV)
    
    img  = thrsImg
    img1 = thrsImg
    img2 = thrsImg
    img3 = thrsImg
    # plt.imshow(img,cmap="gray")
    # plt.show()

    # to focus on foreground instead of background
    kernalLine = np.ones((5,85),np.uint8)

    kernalWord = np.ones((11,21),np.uint8)

    kernalChar = np.ones((),np.uint8)

    imgLine = cv2.dilate(img,kernalLine,iterations=1)
    # plt.imshow(imgLine)
    # plt.show()
    imgWord = cv2.dilate(img,kernalWord,iterations=1)
    # plt.imshow(imgWord)
    # plt.show()
    imgChar = cv2.dilate(img,kernalChar,iterations=1)
    # plt.imshow(imgChar,cmap='gray')
    # plt.show()

    # RETR_EXTERNAL for getting external contours ony and CHAIN_APPROX_NONE for not getting all points of contours (if img is not straight then if don't want all points) but some points only
    contoursLine , heirarchy = cv2.findContours(imgLine,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

    sorted_contours_line = sorted(contoursLine,key=lambda ctr : cv2.boundingRect(ctr)[1])

    for line in sorted_contours_line:
        
        x,y,w,h = cv2.boundingRect(line)
        requiredPart = imgWord[y:y+h,x:x+w]
        # print("line")
        # print(x,y,w,h)
        
        # cv2.rectangle(img1,(x,y),(x+w,y+h),(0, 0, 0),2)
        # plt.imshow(img1)
        # plt.show()

        # find contours for words from dailated word image
        contoursWord , heirarchy = cv2.findContours(requiredPart,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        sorted_contours_word = sorted(contoursWord,key=lambda ctr : cv2.boundingRect(ctr)[0])

        for word in sorted_contours_word:
            # print(cv2.contourArea(word))

            if(cv2.contourArea(word) <300):
                continue
            xw,yw,ww,hw = cv2.boundingRect(word)
            # print("word")

            # print(xw,yw,ww,hw)


            requiredPartWord = imgChar[y:y+h,x+xw:x+xw+ww]
            # cv2.rectangle(img2,(x+xw,y),(x+xw+ww,y+h),(255, 0, 0),2)
           

            contoursChar , heirarchy = cv2.findContours(requiredPartWord,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            sorted_contours_char = sorted(contoursChar,key=lambda ctr : cv2.boundingRect(ctr)[0])
            wordsChar =[]
            for char in sorted_contours_char:
                # print(cv2.contourArea(char))
                
                if(cv2.contourArea(char) < 60):
                    continue
                xc,yc,wc,hc = cv2.boundingRect(char)
                # print("char")

                # print(xw,yw,ww,hw)
                wordsChar.append(imgCopy[y:h+y,xc+x+xw:xw+xc+x+wc])
                # plt.imshow(imgCopy[y:h+y,xc+x+xw:xw+xc+x+wc])
                # plt.show()
                cv2.rectangle(img3,(xc+x+xw,y),(xc+x+xw+wc,h+y),(255, 0, 0),2)
                # plt.imshow(img3)
                # plt.show()
            chars.append(wordsChar)
    # plt.imshow(img3)
    # plt.show()
    # plt.imshow(imgCopy)
    # plt.show()
    return chars










