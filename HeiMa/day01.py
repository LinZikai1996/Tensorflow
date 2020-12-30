import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 实现一个加法运算
# 修改 op name
a = tf.constant(10, name="a")
b = tf.constant(20, name="b")
c = tf.add(a, b, name="sum_c")

print("打印 a: \n", a)
print("打印 b: \n", b)
print("打印 c: \n", c)

# 打印所有默认图（地址对象）
print(a.graph)
print(b.graph)
print(c.graph)

# 定义一个占位符
# 在运行时填充数据
d = tf.compat.v1.placeholder(dtype=tf.float32, name="d")
e = tf.compat.v1.placeholder(dtype=tf.float32, name="e")
f = tf.add(d, e, name="sum_f")

# 获取默认图
g = tf.compat.v1.get_default_graph()
print("获取当前加法运算", g)

# 创建另一张图
new_g = tf.Graph()
with new_g.as_default():
    new_a = tf.constant(20)
    new_b = tf.constant(30)
    new_c = tf.add(new_a, new_b)

print(new_a.graph)
print(new_b.graph)
print(new_c.graph)

con_1 = tf.constant(3.0)
con_2 = tf.constant([1, 2, 3, 4])
con_3 = tf.constant([[1, 2], [3, 4]])
con_4 = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

print(con_1.shape, con_2.shape, con_3.shape, con_4.shape)

print(tf.ones([3, 5]))
print(tf.zeros([3, 5]))
print(tf.random.normal([3, 5], mean=0.0, stddev=1.0))

# 形状变化
con_5 = tf.compat.v1.placeholder(tf.float32, shape=[None, None])
con_6 = tf.compat.v1.placeholder(tf.float32, shape=[None, 10])
con_7 = tf.compat.v1.placeholder(tf.float32, shape=[2, 3])

print("con_5 shape:", con_5.get_shape())
print("con_6 shape:", con_6.get_shape())
print("con_7 shape:", con_7.get_shape())

# 静态形状
# 对于张量本身已经固定的不能再次修改
con_5.set_shape([5, 6])
print("con_5 shape:", con_5.get_shape())

# 指定一个会话运行程序
# 会话只运行默认的那张图，如果有多个图，需要有多个会话
# 打印设备信息, config=tf.ConfigProto(allow_soft_placement=True,log_device_placement=True)
with tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True,
                                                          log_device_placement=True)) as sess:
    print(sess.graph)
    # 1. 写入events文件中去
    # file_writer = tf.summary.FileWriter("./tmp/summary", graph=sess.graph)
    # 2. tensorboard --logdir=E:\project\python\Tensorflow\HeiMa\tmp\summary

    # run()
    c_res = sess.run(c)
    print(c_res)
    result_a, result_b, result_c = sess.run([a, b, c])
    print("多输出结果: \n", result_a, result_b, result_c)
    print("占位符结果: \n", sess.run(f, feed_dict={d: 30.0, e: 40.0}))
