import tensorflow as tf

mnist =  tf.keras.datasets.mnist
(x_train,y_train), (x_test,y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)  # scales data between 0 and 1
x_test = tf.keras.utils.normalize(x_test, axis=1)  # scales data between

# x_train = tf.keras.utils.normalize(x_train, axis=1).reshape(x_train.shape[0], -1)
# x_test = tf.keras.utils.normalize(x_test, axis=1).reshape(x_test.shape[0], -1)

model = tf.keras.models.Sequential()

model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=3)

val_loss, val_acc = model.evaluate(x_test, y_test)

print(val_acc,val_loss)


model.save('epic_num_reader.model')

new_model = tf.keras.models.load_model('epic_num_reader.model')
predictions = new_model.predict(x_test)





# import matplotlib as plt
#
# plt.imshow(x_train[0])
# plt.show()

import matplotlib.pyplot as plt

plt.imshow(x_train[0])
plt.show()
# plt.imshow(x_test[0],cmap=plt.cm.binary)
# plt.show()

import numpy as np
print(np.argmax(predictions[0]))
# print(x_train)


# import input_data
# mnist = input_data.read_data_sets('/data', one_hot=True)
# import  tensorflow as tf
#
# learning_rate = 0.01
# training_interaction = 3
# batch_size =100
# display_step = 2
#
#
# x = tf.placeholder("float",[None,784])
# y = tf.placeholder("float",[None,10])
#
# W = tf.Variable(tf.zeros([784,10]))
# b = tf.Variable(tf.zeros([10]))
# with tf.name_scope("Wx_b") as scope:
#     # Construct a linear model
#     model = tf.nn.softmax(tf.matmul(x, W) + b)  # Softmax
#
# # Add summary ops to collect data
# w_h = tf.summary.histogram("weights", W)
# b_h = tf.summary.histogram("biases", b)
#
#
# with tf.name_scope("cost_function") as scope:
#
#     cost_function = -tf.reduce_sum(y*tf.log(model))
#
#     tf.summary.scalar("cost_function", cost_function)
#
# with tf.name_scope("train") as scope:
#     optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)
#     init = tf.initialize_all_variables()
#     merged_summary_op = tf.summary.merge_all()
#
# with tf.Session() as sess:
#     sess.run(init)
#
#     # Change this to a location on your computer
#     summary_writer = tf.summary.FileWriter('data/logs', graph_def=sess.graph_def)
#
#     # Training cycle
#     for iteration in range(training_iteration):
#         avg_cost = 0.
#         total_batch = int(mnist.train.num_examples / batch_size)
#         # Loop over all batches
#         for i in range(total_batch):
#             batch_xs, batch_ys = mnist.train.next_batch(batch_size)
#             # Fit training using batch data
#             sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
#             # Compute the average loss
#             avg_cost += sess.run(cost_function, feed_dict={x: batch_xs, y: batch_ys}) / total_batch
#             # Write logs for each iteration
#             summary_str = sess.run(merged_summary_op, feed_dict={x: batch_xs, y: batch_ys})
#             summary_writer.add_summary(summary_str, iteration * total_batch + i)
#         # Display logs per iteration step
#         if iteration % display_step == 0:
#             print("Iteration:", '%04d' % (iteration + 1), "cost=", "{:.9f}".format(avg_cost))
#
#     print("Tuning completed!")
#
#     # Test the model
#     predictions = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
#     # Calculate accuracy
#     accuracy = tf.reduce_mean(tf.cast(predictions, "float"))
#     print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
