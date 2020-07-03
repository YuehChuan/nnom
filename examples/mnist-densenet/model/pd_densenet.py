'''
    Copyright (c) 2020
    YuehChuan, Power Diagnose Service
    schwarmcyc@pdservice.com
    SPDX-License-Identifier: Apache-2.0
    Change Logs:
    Date           Author       Notes
    2020-07-01     YuehChuan   The first version
'''


import matplotlib.pyplot as plt
import os
import sys
nnscript = os.path.abspath('../../../scripts')
sys.path.append(nnscript)

from keras.models import Sequential, load_model
from keras.models import Model
from keras.datasets import mnist
from keras.datasets import cifar10 #test
from keras.layers import *
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint

from nnom_utils import *

"""
Dataset part
"""
import numpy as np
import cv2
from tqdm import tqdm
import random
from sklearn.model_selection import train_test_split


model_name = 'pd_model.h5'
save_dir = model_name #os.path.join(os.getcwd(), model_name)

CATEGORIES = ["corona", "external", "internal", "noise"]
model_name = 'pd_model.h5'
save_dir = model_name #os.path.join(os.getcwd(), model_name)

DATADIR = '/home/corleone/resnet-torch/kill-the-bits/sample/train/'
IMG_SIZE=224


def create_training_data():
    training_data = []
    for category in CATEGORIES:  #

        path = os.path.join(DATADIR, category)  # create path
        class_num = CATEGORIES.index(category)  # get the classification  (0-3).


        for img in tqdm(os.listdir(path)):  # iterate over each image per dogs and cats
            try:
                img_array = cv2.imread(os.path.join(path, img))  # convert to array
                new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize to normalize data size
                training_data.append([new_array, class_num])  # add this to our training_data
            except Exception as e:  # in the interest in keeping the output clean...
                pass

    return training_data



def dense_block(x, k):
    x1 = Conv2D(k, kernel_size=(3, 3), strides=(1,1), padding="same")(x)
    x1 = BatchNormalization()(x1)
    x1 = ReLU()(x1)

    x2 = concatenate([x, x1],axis=-1)
    x2 = Conv2D(k, kernel_size=(3, 3), strides=(1,1), padding="same")(x2)
    x2 = BatchNormalization()(x2)
    x2 = ReLU()(x2)

    x3 = concatenate([x, x1, x2],axis=-1)
    x3 = Conv2D(k, kernel_size=(3, 3), strides=(1,1), padding="same")(x3)
    x3 = BatchNormalization()(x3)
    x3 = ReLU()(x3)

    x4 = concatenate([x, x1, x2, x3],axis=-1)
    x4 = Conv2D(k, kernel_size=(3, 3), strides=(1,1), padding="same")(x4)
    x4 = BatchNormalization()(x4)
    x4 = ReLU()(x4)
    
    x5 = concatenate([x, x1, x2, x3, x4], axis=-1)
    return x5

def train(x_train, y_train, x_test, y_test, batch_size= 64, epochs = 100):

    inputs = Input(shape=x_train.shape[1:])
    x = Conv2D(12, kernel_size=(5, 5), strides=(1, 1), padding='same')(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPool2D((2,2), strides=(2,2))(x)

    # dense block 1
    x = dense_block(x, k=12)

    # bottleneck
    x = Conv2D(36, kernel_size=(1, 1), strides=(1, 1), padding="same")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPool2D((2, 2), strides=(2, 2))(x)

    # dense block 2
    x = dense_block(x, k=12)

    x = Conv2D(48, kernel_size=(3, 3), strides=(1, 1), padding="same")(x)
    x = Dropout(0.3)(x)
    x = Flatten()(x)
    x = Dense(4)(x)

    predictions = Softmax()(x)

    model = Model(inputs=inputs, outputs=predictions)

    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.summary()

    # save best
    checkpoint = ModelCheckpoint(filepath=save_dir,
            monitor='val_acc',
            verbose=0,
            save_best_only='True',
            mode='auto',
            period=1)
    callback_lists = [checkpoint]

    history = model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              verbose=2,
              validation_data=(x_test, y_test),
              shuffle=True, callbacks=callback_lists)

    # free the session to avoid nesting naming while we load the best model after.
    del model
    K.clear_session()
    return history

def main(weights='weights.h'):
    epochs = 5 # reduced for CI
    num_classes = 4

    # select different dataset as you wish
    dataset = 'pd'
    #dataset = 'cifar'
    if(dataset in 'mnist'):
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        # add channel dimension for mnist data
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

    elif (dataset in 'pd'):
        for category in CATEGORIES:  # do dogs and cats
            path = os.path.join(DATADIR, category)  # create path to dogs and cats
            for img in os.listdir(path):  # iterate over each image per dogs and cats
                img_array = cv2.imread(os.path.join(path, img))
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
                plt.imshow(img_array)  # graph it
                plt.show()  # display!
                break  # we just want one for now so break

        training_data = create_training_data()
        print("finish create data!!!\n")
        print(len(training_data))
        random.shuffle(training_data)
        for sample in training_data[:10]:
            print(sample[1])

        X = []
        Y = []

        for features, label in training_data:
            X.append(features)
            Y.append(label)

        X = np.asarray(X)
        Y = np.asarray(Y)
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.33)

        # add channel dimension for pd data
        x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 3).astype('float32')
        x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 3).astype('float32')
        print(x_train[0].reshape(-1, IMG_SIZE, IMG_SIZE, 1))
    else:
        (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    #convert to ndarray
    x_test = np.asarray(x_test)
    y_test = np.asarray(y_test)
    x_train = np.asarray(x_train)
    y_train = np.asarray(y_train)


    # quantize the range to 0~1
    x_test = x_test.astype('float32')/255
    x_train = x_train.astype('float32')/255
    print("data range", x_test.min(), x_test.max())

    if(os.getenv('NNOM_TEST_ON_CI') == 'YES'):
        shift_list = eval(open('.shift_list').read())
        rP = 0.0
        for i,im in enumerate(x_test):
            X = im.reshape(1,224,224,3)
            f2q(X, shift_list['input_1']).astype(np.int8).tofile('tmp/input.raw')
            if(0 == os.system('./mnist > /dev/null')):
                out = q2f(np.fromfile('tmp/Softmax1.raw',dtype=np.int8),7)
                out = np.asarray(out)
                num, prop = out.argmax(), out[out.argmax()]
                rnum = y_test[i].argmax()
                if(rnum == num):
                    #print('test image %d is %d, predict correctly with prop %s'%(i, rnum, prop))
                    rP += 1.0
                if((i>0) and ((i%1000)==0)):
                    print('%.1f%%(%s) out of %s is correct predicted'%(rP*100.0/i, rP, i))
        print('%.1f%%(%s) out of %s is correct predicted'%(rP*100.0/i, rP, i))
        if(rP/i > 0.8):
            return
        else:
            raise Exception('test failed, accuracy is %.1f%% < 80%%'%(rP*100.0/i))

    # generate binary
    if(not os.path.exists(dataset+'_test_data.bin')):
        # recover the range to 0~127 for MCU
        generate_test_bin(x_test*127, y_test, name=dataset+'_test_data.bin')

    # train model
    if(not os.path.exists(save_dir)):
        history = train(x_train,y_train, x_test, y_test, batch_size=128, epochs=epochs)
        acc = history.history['acc']
        val_acc = history.history['val_acc']
        if(os.getenv('NNOM_ON_CI') == None):
            plt.plot(raload_modelnge(0, epochs), acc, color='red', label='Training acc')
            plt.plot(range(0, epochs), val_acc, color='green', label='Validation acc')
            plt.title('Training and validation accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()
  
    # get best model
    model = load_model(save_dir)

    # evaluate
    evaluate_model(model, x_test, y_test)

    # convert to model on nnom
    generate_model(model, x_test[:100], name=weights)

    return model,x_train,y_train,x_test,y_test

if __name__ == "__main__":
    main()
