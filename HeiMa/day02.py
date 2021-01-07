import tensorflow as tf
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
from tensorflow.python import keras


def main():
    # 导入图片  load_img(路径,指定大小)
    image = load_img("./tmp/battleship/battleship_002.jpg", target_size=(300, 300))
    print(image)

    # 将图片转化为数组
    image = img_to_array(image)
    # 每个像素点0 - 255
    print(image.shape)
    print(image)

    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar100.load_data()
    print(x_train.shape)
    print(y_test.shape)

    (x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()
    print(x_train.shape)
    print(y_test.shape)

    from tensorflow.python.keras.models import Sequential
    from tensorflow.python.keras.layers import Dense, Flatten

    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation=tf.nn.relu),
        Dense(10, activation=tf.nn.softmax)
    ])

    print(model.compile(optimizer=tf.keras.optimizers.Adam(),
                        loss='sparse_categorical_crossentropy',
                        metrics=['accuracy']))


if __name__ == '__main__':
    main()
