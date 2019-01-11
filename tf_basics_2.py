#Variables and Placeholders
import tensorflow as tf
sess = tf.InteractiveSession()

my_tensor = tf.random_uniform((4,4), 0, 1)
my_var = tf.Variable(initial_value=my_tensor)
#sess.run(my_var) is not goint to work. init is needed

init = tf.global_variables_initializer()
sess.run(init)
sess.run(my_var)

ph = tf.placeholder(tf.float32) #Usually (tf.float32, shape=(None,4))
