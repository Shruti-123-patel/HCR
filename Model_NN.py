import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import numpy as np

def NN_model(x_train,x_test,y_train,y_test,sizeImage):
    x_train = (np.array(x_train)).reshape(np.array(x_train).shape[0], sizeImage, sizeImage, 1)
    x_test = (np.array(x_test)).reshape(np.array(x_test).shape[0], sizeImage, sizeImage, 1)
    num_classes = 62
    inputShape = (sizeImage,sizeImage,1)
    batch_size = 256
    learning_rate = 0.0001
    epochs = 10


    # make sequential model
    model = Sequential()
    # add convolution layer 
    model.add(Conv2D(32, kernel_size=(3, 3),activation='relu',input_shape=inputShape))
    # add another convolution layer
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # for ANN we need it
    model.add(Flatten())

    model.add(Dense(256, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.sparse_categorical_crossentropy,optimizer=keras.optimizers.Adam(learning_rate = learning_rate),metrics=['accuracy'])
    
    hist = model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose=3,validation_data=(x_test, y_test))

    score = model.evaluate(x_test, y_test)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])