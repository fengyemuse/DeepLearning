import tensorflow as tf

tf.compat.v1.disable_eager_execution()

x = tf.compat.v1.placeholder(tf.float32, name='x')
y = tf.compat.v1.placeholder(tf.float32, name='y')
z = tf.compat.v1.add(x, y, name='sum')
values = {x: 5.0, y: 4.0}
sess = tf.compat.v1.Session()

summary_writer = tf.compat.v1.summary.FileWriter('/tmp/1', sess.graph)
summary_writer.close()
# print(sess.run([z], values))
