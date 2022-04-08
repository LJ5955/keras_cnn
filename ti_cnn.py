import tensorflow.keras as keras

def build_model():
    inputdata = keras.Input(shape=(499, 39, 1))

    final = keras.layers.Conv2D(32, (3, 3), padding="same",activation='relu')(inputdata)
    final = keras.layers.BatchNormalization()(final)
    final = keras.layers.ReLU()(final)
    final = keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(final)

    final = keras.layers.Conv2D(16, (3, 3), padding="same",activation='relu')(final)
    final = keras.layers.BatchNormalization()(final)
    final = keras.layers.ReLU()(final)
    final = keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(final)

    final = keras.layers.Conv2D(8, (3, 3), padding="same",activation='relu')(final)
    final = keras.layers.BatchNormalization()(final)
    final = keras.layers.ReLU()(final)
    final = keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(final)

    final = keras.layers.Conv2D(16, (3, 3), padding="same",activation='relu')(final)
    final = keras.layers.BatchNormalization()(final)
    final = keras.layers.ReLU()(final)

    final = keras.layers.Flatten()(final)
    final = keras.layers.Dense(3)(final)# 二分类时为2，四分类时为4
    # final = keras.layers.BatchNormalization()(final)
    final = keras.layers.Softmax()(final)

    model = keras.Model(inputs=inputdata, outputs=final)
    optimizer = keras.optimizers.Adam(
        lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None)
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])
    return model






import numpy as np
if __name__ == '__main__':
    model = build_model()

    X_train = np.random.rand(32,499,39,1)
    Y_train = np.random.rand(32,3)
    model.fit(X_train, Y_train, epochs=20, batch_size=32)