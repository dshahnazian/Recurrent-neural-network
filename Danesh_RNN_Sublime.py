import tensorflow as tf
import numpy as np
sess = tf.Session()

x_batch = tf.placeholder(tf.float32, shape=[4, 6, 7])
y_batch = tf.placeholder(tf.float32, shape=[4, 6, 8])


x_list = []

W2 = tf.Variable(np.random.rand(7,15)*0.01, dtype=tf.float32)
b_hidden = tf.Variable(tf.zeros([15]))

W3 = tf.Variable(np.random.rand(15,15)*0.01, dtype=tf.float32)
context = tf.Variable(tf.zeros([4, 15]), dtype=tf.float32)

W4 = tf.Variable(np.random.rand(15,8)*0.01, dtype=tf.float32)
b_output = tf.Variable(tf.zeros([8]))

cross_entropy_sum = tf.Variable(tf.zeros([1]))
accuracy_list = []

for seriesStep in range (6):
    x = tf.squeeze(tf.slice(x_batch, [0,seriesStep,0], [-1, 1, -1]))
    x_list.append(x)
    y_label = tf.squeeze(tf.slice(y_batch, [0,seriesStep,0], [-1, 1, -1]))
    
    hidden = tf.matmul(x,W2) + tf.matmul(context,W3) + b_hidden
    
    # context = hidden
    hidden_clipped = tf.nn.tanh(hidden)    
    context = hidden_clipped
    output = tf.matmul(hidden_clipped, W4) + b_output
    
    y_predicted = tf.nn.softmax(output)
    
    correct_prediction = tf.equal(tf.argmax(y_predicted,1), tf.argmax(y_label,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    accuracy_list.append(accuracy)
        
    cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_label * tf.log(y_predicted), reduction_indices=[1]))
    cross_entropy_sum = cross_entropy_sum + cross_entropy

context_final = context
cross_final = cross_entropy_sum
LEARNING_RATE = 0.5
training = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy_sum)

x_final = x_list
accuracy_final = accuracy_list


    

init = tf.global_variables_initializer()
sess.run(init)

import numpy as np

Start=     [1,0,0,0,0,0,0]
Coffee=    [0,1,0,0,0,0,0];
Tea=       [0,0,1,0,0,0,0];
Water=     [0,0,0,1,0,0,0];
Cream=     [0,0,0,0,1,0,0];
Sugar=     [0,0,0,0,0,1,0];
Stir=      [0,0,0,0,0,0,1];


# %%% output; this is the teacher output 
Coffeeout=  [1,0,0,0,0,0,0,0];          
Teaout=     [0,1,0,0,0,0,0,0];
Waterout=   [0,0,1,0,0,0,0,0];
Creamout=   [0,0,0,1,0,0,0,0];
Sugarout=   [0,0,0,0,1,0,0,0];
Stirout=    [0,0,0,0,0,1,0,0];
Coffeebev=  [0,0,0,0,0,0,1,0];
Teabev=     [0,0,0,0,0,0,0,1];

inputBatch = []
outputBatch = []

# %%% Seq1  tea water second 

TWInp=[Start,Tea,Water,Stir,Sugar,Stir]
TWOut=[Teaout,Waterout,Stirout,Sugarout,Stirout,Teabev]

inputBatch.append(TWInp)
outputBatch.append(TWOut)

# %%% Seq2  tea water second 

TSInp=[Start,Tea,Sugar,Stir,Water,Stir]
TSOut=[Teaout,Sugarout,Stirout,Waterout,Stirout,Teabev]

inputBatch.append(TSInp)
outputBatch.append(TSOut)


# %%% Seq3  coffee water first 

CWInp=[Start,Coffee,Water,Stir,Cream,Stir]
CWOut=[Coffeeout,Waterout,Stirout,Creamout,Stirout,Coffeebev]

inputBatch.append(CWInp)
outputBatch.append(CWOut)


# %%% Seq4  coffee water second

CCInp=[Start,Coffee,Cream,Stir,Water,Stir]
CCOut=[Coffeeout,Creamout,Stirout,Waterout,Stirout,Coffeebev]

inputBatch.append(CCInp)
outputBatch.append(CCOut)




TRAIN_STEPS = 300
weightList1 = []
weightList2 = []
weightList3 = []
weightList4 = []
# with sess.as_default():
for i in range(TRAIN_STEPS):
	x_train = inputBatch
	y_train = outputBatch

	sess.run(training, feed_dict={x_batch: x_train, y_batch: y_train})

	if (i%3==0):
		print('Training Step: ' + str(i) + '  Accuracy =  ' + str(sess.run(accuracy_final, feed_dict={x_batch: x_train, y_batch: y_train})) + '  Loss = ' + str(sess.run(cross_final, {x_batch: x_train, y_batch: y_train})))
		
		b = sess.run(W3)
		weightList1.append(b[0])
		weightList2.append(b[1])

		c = sess.run(W4)
		weightList3.append(c[2])
		weightList4.append(c[3])
		# print(b[:,6:8])

	context = tf.Variable(tf.zeros([4, 15]), dtype=tf.float32)

import matplotlib.pyplot as plt 
plt.figure(1)
plt.subplot(411)
plt.plot(weightList1)
plt.subplot(412)
plt.plot(weightList2)
plt.subplot(413)
plt.plot(weightList3)
plt.subplot(414)
plt.plot(weightList4)
plt.ylabel('some numbers')
plt.show()