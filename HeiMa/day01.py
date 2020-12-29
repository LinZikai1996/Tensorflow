import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 实现一个加法运算
a = tf.constant(10)
b = tf.constant(20)

c = tf.add(a, b)

# 获取默认图
g = tf.compat.v1.get_default_graph()
print("获取当前加法运算", g)

# 打印所有默认图（地址对象）
print(a.graph)
print(b.graph)
print(c.graph)

# 创建另一张图
new_g = tf.Graph()
with new_g.as_default():
    new_a = tf.constant(20)
    new_b = tf.constant(30)
    new_c = tf.add(new_a, new_b)

print(new_a.graph)
print(new_b.graph)
print(new_c.graph)

# 指定一个会话运行程序
# 会话只运行默认的那张图，如果有多个图，需要有多个会话
with tf.compat.v1.Session() as sess:
    print(sess.graph)
    # 1. 写入events文件中去
    # file_writer = tf.summary.FileWriter("./tmp/summary", graph=sess.graph)
    # 2. tensorboard --logdir=E:\project\python\Tensorflow\HeiMa\tmp\summary
    c_res = sess.run(c)
    print(c_res)
