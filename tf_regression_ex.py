import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

x_data = np.linspace(0,10,1e6)
noise = np.random.randn(len(x_data))

# y = mx+b
# b = 5

y_true = (0.5 * x_data) + 5 + noise

x_df = pd.DataFrame(data=x_data, columns=['X Data'])
y_df = pd.DataFrame(data=y_true, columns=['Y'])

# x_df.head() #to show just the beginnig
# x_df.tail() #to show just the ending

my_data = pd.concat([x_df, y_df],axis=1)
my_data.sample(n=250).plot(kind='scatter', x='X Data', y='Y')
# plt.show()

batch_size = 8
m = tf.Variable(0.81) #random numbers
b = tf.Variable(0.17) #random numbers

xph = tf.placeholder(tf.float32, [batch_size])
yph = tf.placeholder(tf.float32, [batch_size])

y_model = m*xph + b

error = tf.reduce_sum(tf.square(yph-y_model))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
train = optimizer.minimize(error)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    batches = 1000
    for i in range(batches):
        rand_ind = np.random.randint(len(x_data), size=batch_size)
        feed = {xph:x_data[rand_ind], yph:y_true[rand_ind]}
        sess.run(train, feed_dict=feed)
    model_m, model_b = sess.run([m,b])

print('m sendo perto de 0.5: {}'.format(model_m))
print('b sendo perto de 5: {}'.format(model_b))

y_hat = x_data*model_m + model_b

plt.plot(x_data, y_hat, 'r')
# plt.show()

#Estimators
feat_cols = [tf.feature_column.numeric_column('x', shape=[1])] #has to be a list
estimator = tf.estimator.LinearRegressor(feature_columns=feat_cols)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_data, y_true,
                                    test_size=0.3, random_state=101)

input_func = tf.estimator.inputs.numpy_input_fn({'x':x_train}, y_train,
                                    batch_size=8, num_epochs=None, shuffle=True)

train_input_func = tf.estimator.inputs.numpy_input_fn({'x':x_train}, y_train,
                                    batch_size=8, num_epochs=1000, shuffle=False)

test_input_func = tf.estimator.inputs.numpy_input_fn({'x':x_test}, y_test,
                                    batch_size=8, num_epochs=1000, shuffle=False)

estimator.train(input_fn=input_func, steps=1000)

train_metrics = estimator.evaluate(input_fn=train_input_func, steps=1000)

tests_metrics = estimator.evaluate(input_fn=test_input_func, steps=1000)

print('TRAINING DATA METRICS')
print(train_metrics)

print('TESTING DATA METRICS')
print(tests_metrics)

brand_new_data = np.linspace(0,10,10)
input_fn_predict = tf.estimator.inputs.numpy_input_fn({'x':brand_new_data},
                                                    shuffle=False)
list( estimator.predict(input_fn=input_fn_predict) )

predictions = []
for pred in estimator.predict(input_fn=input_fn_predict):
    predictions.append(pred['predictions'])

plt.figure()
my_data.sample(n=250).plot(kind='scatter', x='X Data', y='Y')
plt.plot(brand_new_data, predictions, 'r*')
plt.show()
