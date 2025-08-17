import keras
from keras.models import Sequential
from keras.layers import Dense,Flatten
import numpy as np
from keras.applications import VGG16
import pickle
import tensorflow as tf

def vggNet(x_train,x_test,y_train,y_test,img_size):
    x_train = np.array(x_train)
    x_test = np.array(x_test)
    x_train = (np.repeat(x_train[...,np.newaxis],3,-1))
    x_test = (np.repeat(x_test[...,np.newaxis],3,-1))
    print(y_train)
    num_classes = 62
    inputShape = (img_size,img_size,3)
    batch_size = 256
    learning_rate = 0.0001
    epochs = 10

    base_model = VGG16(weights = "imagenet", include_top=False, input_shape = inputShape)
    base_model.summary()
    model = Sequential()

    model.add(base_model)

    
    model.add(Flatten())

    model.add(Dense(256, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adam(learning_rate = learning_rate),metrics=['accuracy'])
    
    # hist = model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_test, y_test))

    # filename = 'vggNetModelCnn.sav'


    checkpoint_filepath = 'newAugmentedPractice.sav'
    # model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    #   filepath=checkpoint_filepath,
    #   save_weights_only=True,
    #   monitor='val_accuracy',
    #   mode='max',
    #   save_best_only=True)



    # pickle.dump(model, open(filename, 'wb'))

    # with open(checkpoint_filepath,'rb') as f:
    #     model = pickle.load(f)
    model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_test, y_test))
    pickle.dump(model, open(checkpoint_filepath, 'wb'))
    score = model.evaluate(x_test, y_test)
    
    predict_x = model.predict(x_test)
    y_pred = []

    for i in predict_x :
       y_pred.append(np.argmax(i))
    print(y_pred)
    print(y_test)
    
    
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    