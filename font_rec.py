import keras
from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D , UpSampling2D ,Conv2DTranspose

def create_model():
    model = Sequential()

    model.add(Conv2D(64, kernel_size=(48, 48), activation='relu', input_shape=(105,105,1)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, kernel_size=(24, 24), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2DTranspose(128, (24,24), strides = (2,2), activation = 'relu', padding='same', kernel_initializer='uniform'))
    model.add(UpSampling2D(size=(2, 2)))

    model.add(Conv2DTranspose(64, (12,12), strides = (2,2), activation = 'relu', padding='same', kernel_initializer='uniform'))
    model.add(UpSampling2D(size=(2, 2)))

    model.add(Conv2D(256, kernel_size=(12, 12), activation='relu'))
    model.add(Conv2D(256, kernel_size=(12, 12), activation='relu'))
    model.add(Conv2D(256, kernel_size=(12, 12), activation='relu'))
    model.add(Flatten())

    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(4096,activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(2383,activation='relu'))
    model.add(Dense(5, activation='softmax'))

    return model

class FontRecognizer:
    def __init__(self, args=None):
        self.predictor = create_model()