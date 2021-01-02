import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def linear_regression():
    """
    Tensorflow 实现线性回归
    :return:
    """

    # 1. 准备数据集
    # y = 0.8x +0.7
    # x[100, 1] * 0.8 + 0.7 = y[100, 1]

    # 使用正态分布生成数据
    with tf.variable_scope("original data"):
        X = tf.random.normal([100, 1], mean=0.0, stddev=1, name="original_data_x")
        Y_true = tf.matmul(X, [[0.8]]) + [[0.7]]

    # 2. 建立线性模型
    # trainable 指定参数不训练
    with tf.variable_scope("linear model"):
        weight = tf.Variable(initial_value=tf.random.normal([1, 1]), name="weight")
        bias = tf.Variable(initial_value=tf.random.normal([1, 1]), name="bias")

        # 预测结果
        Y_predict = tf.matmul(X, weight) + bias

    # 3. 均方误差
    with tf.variable_scope("loss"):
        loss = tf.reduce_mean(tf.square(Y_true - Y_predict))

    # 4. 梯度下降
    with tf.variable_scope("optimizer"):

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(loss)

    # 收集张量
    tf.summary.scalar('losses', loss)
    tf.summary.histogram('weight', weight)
    tf.summary.histogram('bias', bias)

    merge = tf.summary.merge_all()
    # 初始化op
    init_op = tf.compat.v1.global_variables_initializer()

    with tf.compat.v1.Session() as sess:
        sess.run(init_op)
        for i in range(100):
            print("--------------------------------------------------------------")
            sess.run(optimizer)
            sess.run(loss)
            summany = sess.run(merge)

            print(sess.run([weight, bias]))
    return None


if __name__ == '__main__':
    linear_regression()
