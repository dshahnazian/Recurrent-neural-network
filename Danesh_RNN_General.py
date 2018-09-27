import tensorflow as tf
import numpy as np





truncatedBackpropLength = 5
numOfInputNodes = 1
numOfOutputNodes = 2

numOfHiddenNodes = 6

batchSize = 4

randScaler = 0.01



sess = tf.Session()

x_batch = tf.placeholder(tf.float32, shape=[None, truncatedBackpropLength, numOfInputNodes])
y_batch = tf.placeholder(tf.float32, shape=[None, truncatedBackpropLength, numOfOutputNodes])


x_list = []

W2 = tf.Variable(np.random.rand(numOfInputNodes,numOfHiddenNodes)*randScaler, dtype=tf.float32)
b_hidden = tf.Variable(tf.zeros([numOfHiddenNodes]))

W3 = tf.Variable(np.random.rand(numOfHiddenNodes,numOfHiddenNodes)*randScaler, dtype=tf.float32)
context = tf.Variable(tf.zeros([batchSize, numOfHiddenNodes]), dtype=tf.float32)

W4 = tf.Variable(np.random.rand(numOfHiddenNodes,numOfOutputNodes)*randScaler, dtype=tf.float32)
b_output = tf.Variable(tf.zeros([numOfOutputNodes]))

cross_entropy_sum = tf.Variable(tf.zeros([1]))
accuracy_list = []


for seriesStep in range (truncatedBackpropLength):
    xTemp = tf.squeeze(tf.slice(x_batch, [0,seriesStep,0], [-1, 1, -1]))
    y_label = tf.squeeze(tf.slice(y_batch, [0,seriesStep,0], [-1, 1, -1]))

    if (numOfInputNodes == 1):
    	xTemp = tf.reshape(xTemp, [batchSize, numOfInputNodes])
    	y_label = tf.reshape(y_label, [batchSize, numOfOutputNodes])

    x_list.append(xTemp)


    
    hidden = tf.matmul(xTemp,W2) + tf.matmul(context,W3) + b_hidden
    
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
LEARNING_RATE = 0.05
training = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(cross_entropy_sum)

x_final = x_list
accuracy_final = accuracy_list


    

init = tf.global_variables_initializer()
sess.run(init)

import numpy as np

# Start=     [1,0,0,0,0,0,0]
# Coffee=    [0,1,0,0,0,0,0];
# Tea=       [0,0,1,0,0,0,0];
# Water=     [0,0,0,1,0,0,0];
# Cream=     [0,0,0,0,1,0,0];
# Sugar=     [0,0,0,0,0,1,0];
# Stir=      [0,0,0,0,0,0,1];


# # %%% output; this is the teacher output 
# Coffeeout=  [1,0,0,0,0,0,0,0];          
# Teaout=     [0,1,0,0,0,0,0,0];
# Waterout=   [0,0,1,0,0,0,0,0];
# Creamout=   [0,0,0,1,0,0,0,0];
# Sugarout=   [0,0,0,0,1,0,0,0];
# Stirout=    [0,0,0,0,0,1,0,0];
# Coffeebev=  [0,0,0,0,0,0,1,0];
# Teabev=     [0,0,0,0,0,0,0,1];

# inputBatch = []
# outputBatch = []

# # %%% Seq1  tea water second 

# TWInp=[Start,Tea,Water,Stir,Sugar,Stir]
# TWOut=[Teaout,Waterout,Stirout,Sugarout,Stirout,Teabev]

# inputBatch.append(TWInp)
# outputBatch.append(TWOut)

# # %%% Seq2  tea water second 

# TSInp=[Start,Tea,Sugar,Stir,Water,Stir]
# TSOut=[Teaout,Sugarout,Stirout,Waterout,Stirout,Teabev]

# inputBatch.append(TSInp)
# outputBatch.append(TSOut)


# # %%% Seq3  coffee water first 

# CWInp=[Start,Coffee,Water,Stir,Cream,Stir]
# CWOut=[Coffeeout,Waterout,Stirout,Creamout,Stirout,Coffeebev]

# inputBatch.append(CWInp)
# outputBatch.append(CWOut)


# # %%% Seq4  coffee water second

# CCInp=[Start,Coffee,Cream,Stir,Water,Stir]
# CCOut=[Coffeeout,Creamout,Stirout,Waterout,Stirout,Coffeebev]

# inputBatch.append(CCInp)
# outputBatch.append(CCOut)


total_series_length = 50000
echo_step = truncatedBackpropLength
batch_length = int(total_series_length/batchSize)

def generateData():
	x = np.array(np.random.choice(2, total_series_length, p=[0.5, 0.5]))
	y = np.roll(x, echo_step)
	y[0:echo_step] = 0

	y_hot = [(1-yTemp)*[1,0] + yTemp*[0,1] for yTemp in y]


	y_hot_list = []
	for i in range(batchSize):
		y_hot_Temp = y_hot[i*batch_length:i*batch_length+batch_length]
		y_hot_list.append(y_hot_Temp)

	x = x.reshape((batchSize, batch_length))  # The first index changing slowest, subseries as rows
	# y_hot = y_hot.reshape((batchSize, batch_length))

	return (x, y_hot_list)


x, y = generateData()

print('Size of X: ' + str(len(x)) + '*' + str(len(x[0])))
print('Size of Y: ' + str(len(y)) + '*' + str(len(y[0])))

import random

TRAIN_STEPS = 50000
weightList1 = []
weightList2 = []
weightList3 = []
weightList4 = []
# with sess.as_default():
for i in range(TRAIN_STEPS):
	# randIndex = random.randint(truncatedBackpropLength, len(x[0]))
	randIndex = i+truncatedBackpropLength

	x_train = x[:,randIndex-truncatedBackpropLength:randIndex]
	y_train = [tempRow[randIndex-truncatedBackpropLength:randIndex] for tempRow in y]
	# y[:,randIndex-truncatedBackpropLength:randIndex]

	x_train = x_train.reshape((batchSize, truncatedBackpropLength, 1))
	# y_train = y_train.reshape((batchSize, truncatedBackpropLength, 2))


	sess.run(training, feed_dict={x_batch: x_train, y_batch: y_train})

	if (i%100==0):
		# randIndex = random.randint(truncatedBackpropLength, len(x[0]))
		randIndex = i+truncatedBackpropLength

		x_test = x[:,randIndex-truncatedBackpropLength:randIndex]
		y_test = [tempRow[randIndex-truncatedBackpropLength:randIndex] for tempRow in y]
		# y[:,randIndex-truncatedBackpropLength:randIndex]

		x_test = x_test.reshape((batchSize, truncatedBackpropLength, 1))
		# y_test = y_test.reshape((batchSize, truncatedBackpropLength, 1))

		# print('input: ')
		# print(x_test)

		print('Training Step: ' + str(i) + '  Accuracy =  ' + str(sess.run(accuracy_final, feed_dict={x_batch: x_train, y_batch: y_train})) + '  Loss = ' + str(sess.run(cross_final, {x_batch: x_train, y_batch: y_train})))
		
		b = sess.run(W3)
		weightList1.append(b[0])
		weightList2.append(b[1])

		c = sess.run(W4)
		weightList3.append(c[2])
		weightList4.append(c[3])
		# print(b[:,6:8])

	# context = tf.Variable(tf.zeros([batchSize, numOfHiddenNodes]), dtype=tf.float32)

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