import tensorflow as tf
tf.compat.v1.disable_eager_execution()
hello = tf.compat.v1.constant('hello tensorflow')

sess = tf.compat.v1.Session()
print(sess.run(hello))

# tensorflow2.0没有Session了
