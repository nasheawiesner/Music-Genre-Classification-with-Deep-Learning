import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
import csv
import numpy as np
import random


'''
input > weight > hidden layer 1(activation function) > weights > hidden layer 2(activation function) > weights > output layer

compare output to intended output > cost or loss function (cross entropy)
optimization function (optimizer) > minimize cost (AdamOptimizer...SGD, AdaGrad)

backpropogation

feed forward + backprop = epoch

'''

with open('dataset.csv', 'r') as f:
    reader = csv.reader(f)
    counter = 0
    featureSet = []
    labelSet = []
    for row in reader:
        if counter == 0:
            random.shuffle(row)
        features = []
        labels = []
        for i in range(68,195):
            features.append(float(row[i]))
        features = np.array(features)
        featureSet.append(features)
        #for j in range(197,198 ):#227
        labels.append(row[226])
        if labels[0] == '0':
            labels.append('1')
        elif labels[0] == '1':
            labels.append('0')
        labels = np.array(labels)
        labelSet.append(labels)
        counter += 1

testing_size = int(0.1 * len(features))

train_x = list(featureSet[:-testing_size])
train_y = list(labelSet[:-testing_size])
test_x = list(featureSet[-testing_size:])
test_y = list(labelSet[-testing_size:])

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500
n_nodes_hl4 = 500

n_classes = 2


#height X width
x = tf.placeholder('float',[None,len(train_x[0])])
y = tf.placeholder('float')

batch_size = 50

def neural_network_model(data):

    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([len(train_x[0]), n_nodes_hl1])),
                      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}
    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}
    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}
    hidden_4_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_nodes_hl4])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl4]))}
    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl4, n_classes])),
                      'biases': tf.Variable(tf.random_normal([n_classes]))}

#(input_data * weights) + biases

    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    l4 = tf.add(tf.matmul(l3, hidden_4_layer['weights']), hidden_4_layer['biases'])
    l4 = tf.nn.relu(l4)

    output = tf.matmul(l4, output_layer['weights']) + output_layer['biases']

    return output

def train_neural_network(x):
    prediction = neural_network_model(x)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
    optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

    #cycles feed forward + backprop
    hm_epochs = 10

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            i = 0
            while i < len(train_x):
                start = i
                end = i + batch_size

                batch_x = np.array(train_x[start:end])
                #x = tf.reshape(x, [len(batch_x),len(train_x)])
                batch_y = np.array(train_y[start:end])
                _,c = sess.run([optimizer,cost], feed_dict={x:batch_x,y:batch_y})
                epoch_loss += c
                i += batch_size
            print('Epoch', epoch+1, "completed out of", hm_epochs, 'loss:',epoch_loss)

        correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
        accuracy = tf.reduce_mean(tf.cast(correct,'float'))
        print('Accuracy:',accuracy.eval({x:test_x, y:test_y}))

train_neural_network(x)

