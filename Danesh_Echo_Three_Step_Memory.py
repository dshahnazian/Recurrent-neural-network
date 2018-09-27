from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

num_epochs = 100
total_series_length = 50000
truncated_backprop_length = 15 # Memory Capacity
state_size = 4
num_classes = 2
echo_step = 3
batch_size = 5
num_batches = total_series_length//batch_size//truncated_backprop_length
batch_length = int(total_series_length/batch_size)

def generateData():
	x = np.array(np.random.choice(2, total_series_length, p=[0.5, 0.5]))
	y = np.roll(x, echo_step)
	y[0:echo_step] = 0

	

	x = x.reshape((batch_size, batch_length))  # The first index changing slowest, subseries as rows
	y = y.reshape((batch_size, batch_length))

	return (x, y)


x, y = generateData()


batchX_placeholder = tf.placeholder(tf.float32, [batch_size, truncated_backprop_length])
batchY_placeholder = tf.placeholder(tf.int32, [batch_size, truncated_backprop_length])

init_state = tf.placeholder(tf.float32, [batch_size, state_size])

# State size + 1 is the number of input nodes and state_size is the number of nodes in the hidden layer.
W = tf.Variable(np.random.rand(state_size+1, state_size), dtype=tf.float32)
b = tf.Variable(np.zeros((1,state_size)), dtype=tf.float32)

# W2 is the weights between hidden layer and the output layer
W2 = tf.Variable(np.random.rand(state_size, num_classes),dtype=tf.float32)
b2 = tf.Variable(np.zeros((1,num_classes)), dtype=tf.float32)


# Unpack columns
inputs_series = tf.unstack(batchX_placeholder, axis=1)
labels_series = tf.unstack(batchY_placeholder, axis=1)


# Forward pass
current_state = init_state # Size of (5,4)
states_series = []
for current_input in inputs_series: # current_input is a numpy array and contains 5 members. inputs_series is an array of truncated_backprop_length size which contains batch_size inputs in it.
    current_input = tf.reshape(current_input, [batch_size, 1]) # current_input size is (5,1)
    input_and_state_concatenated = tf.concat(1, [current_input, current_state])  # input_and_state_concatenated size is (5,1) + (5,4) = (5,5)

    activation = tf.matmul(input_and_state_concatenated, W) + b # activation size is (5,4)
    next_state = tf.tanh(activation)  # Instead of Softmax
    states_series.append(next_state)
    current_state = next_state


logits_series = [tf.matmul(state, W2) + b2 for state in states_series] #Broadcasted addition
predictions_series = [tf.nn.softmax(logits) for logits in logits_series]

losses = [tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels) for logits, labels in zip(logits_series,labels_series)]
total_loss = tf.reduce_mean(losses)

	train_step = tf.train.AdagradOptimizer(0.3).minimize(total_loss)




with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    output = sess.run(inputs_series, feed_dict={batchX_placeholder: x[:, 0:15]})
    print(output[0])
