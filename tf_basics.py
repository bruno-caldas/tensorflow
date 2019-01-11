import tensorflow as tf

print(tf.__version__)
hello = tf.constant("Hello")
world = tf.constant("World")
var1 = tf.constant(2)
var2 = tf.constant(3)
matrix_of_7 = tf.fill((3,3), 7)
column_matrix = tf.constant([[1], [2], [3]])

with tf.Session() as sess:
    result = sess.run(hello + world)
    sum =  sess.run(var1+var2)
    mat = sess.run(matrix_of_7)
    multiplication = sess.run(tf.matmul(matrix_of_7, column_matrix) )

print(result)
print('A soma eh: {}'.format(sum))
print('A matriz de setes eh: {}'.format(mat))
print('A multiplicacao eh: {}' .format(multiplication))
