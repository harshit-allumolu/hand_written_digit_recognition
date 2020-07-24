import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.layers import Conv2D, MaxPooling2D 
import numpy as np

# load dataset
def load_dataset():
    # use load_data()
    (x_train,y_train),(x_test,y_test) = mnist.load_data()
    return x_train,y_train,x_test,y_test


# preprocess and normalize data
def preprocess(x_train,y_train,x_test,y_test):
    x_train = keras.utils.normalize(x_train,axis=1)
    x_test = keras.utils.normalize(x_test,axis=1)
    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)
    return x_train,y_train,x_test,y_test

# tensorflow Sequential model
def define_model(input_shape,num_classes):
    #define model
    model = Sequential()
    model.add(Flatten(input_shape = input_shape))
    model.add(Dense(units=128,activation='relu'))
    model.add(Dense(units=128,activation='relu'))
    model.add(Dense(units=num_classes,activation='softmax'))
    
    # compile model
    model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
    return model


# main function
def main():
    # load dataset
    x_train,y_train,x_test,y_test = load_dataset()
    # preprocess
    x_train,y_train,x_test,y_test = preprocess(x_train,y_train,x_test,y_test)
    print("No. of training samples : {}".format(x_train.shape[0]))
    print("No. of test samples : {}".format(x_test.shape[0]))
    # define a model
    input_shape = (28,28)
    num_classes = 10
    model = define_model(input_shape,num_classes)
    # train the model
    history = model.fit(x_train,y_train,epochs=3)
    print("training successful")
    # evaluate the model
    loss, accuracy = model.evaluate(x_test,y_test)
    print("loss : {}\naccuracy : {}".format(loss,accuracy))
    # save the model
    model.save("model.pb")
    print("trained model saved as model.pb")


if __name__ == "__main__":
    main()