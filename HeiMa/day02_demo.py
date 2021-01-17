import tensorflow as tf
from tensorflow.python import keras
from tensorflow.python.keras.layers import Dense, Flatten
import os
import numpy as np

# 构建双层神经网络去训练

class SingleNN(object):
    # 2.建立模型
    model = keras.Sequential([
        # 将输入数据转换成神经网络要求的数据形状
        Flatten(input_shape=(28, 28)),
        # 隐藏层， 128层
        Dense(128, activation=tf.nn.relu),
        # 十个类别的分类问题
        Dense(10, activation=tf.nn.softmax)
    ])

    def __init__(self):
        # 1.读取数据集
        #   返回两元组
        (self.x_train, self.y_train), (self.x_test, self.y_test) = keras.datasets.fashion_mnist.load_data()
        #   数据进行归一化
        self.x_train = self.x_train / 255.0
        self.x_test = self.x_test / 255.0

    def singlenn_compile(self):
        # 3.编译模型优化器、损失、准确率
        SingleNN.model.compile(optimizer=keras.optimizers.SGD(lr=0.01),
                               loss=keras.losses.sparse_categorical_crossentropy,
                               metrics=['accuracy'])

    def singlenn_fit(self):
        # 4.进行训练
        #  batch_size一次训练个数
        #  epochs迭代次数
        SingleNN.model.fit(self.x_train, self.y_train, epochs=200, batch_size=128)
        return None

    def singlenn_evalute(self):
        test_loss, test_acc = SingleNN.model.evaluate(self.x_test, self.y_test)
        print("test_acc: " + str(test_acc) + ", test_loss: " + str(test_loss))
        return None

    def singlenn_save_model(self):
        SingleNN.model.save_weights("./tmp/ckpt/SingleNN")
        return None

    def singlenn_predict(self):
        if os.path.exists("./tmp/ckpt/checkpoint"):
            SingleNN.model.load_weights("./tmp/ckpt/SingleNN")
        return SingleNN.model.predict(self.x_test)


if __name__ == '__main__':
    snn = SingleNN()

    snn.singlenn_compile()
    snn.singlenn_fit()
    snn.singlenn_evalute()
    snn.singlenn_save_model()
    predict = snn.singlenn_predict()
    print(predict)
    print(np.argmax(predict, axis=1))
