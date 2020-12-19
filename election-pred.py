#!/usr/bin/env python
# coding: utf-8

# # election prediction

# In[1]:


import pandas as pd
import numpy as np
import tensorflow.compat.v1 as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
tf.compat.v1.disable_eager_execution()


RANDOMSEED = 40

input_data=pd.read_csv("C:/Users/97254/Downloads/county_factsNew.csv",encoding='latin-1')
input_data=input_data.iloc[50:180, :].sample(frac=1) # all rows, all the features and no labels

input_data.head()


# # labelencoder

# In[2]:


labelencoder=LabelEncoder()
for col in input_data.columns:
    input_data[col] = labelencoder.fit_transform(input_data[col].astype(str))

target = input_data.iloc[80:150, 22].sample(frac=1)# all ows, label only
max1=[]


# # slice data

# In[3]:


def load_data():

    data=input_data.iloc[:200, 3:21] # all rows, all the features and no labels
    target = input_data.iloc[:200, 22]  # all rows, label only

    # Prepend the column of 1s for bias
    L, W  = data.shape
    all_X = np.ones((L, W + 1))
    all_X[:, 1:] = data
    num_labels = len(np.unique(target))
    all_y = np.eye(num_labels)[target]
    return train_test_split(all_X, all_y, test_size=0.33, random_state=RANDOMSEED)


# # model 1

# In[4]:


def initialize_weights(shape, stddev):
    weights = tf.random.normal(shape, stddev=stddev)
    return tf.Variable(weights)

def forward_propagation(X, weights_1, weights_2):
    sigmoid = tf.nn.sigmoid(tf.matmul(X, weights_1))
    y = tf.matmul(sigmoid, weights_2)
    return y

def run(h_size, stddev, sgd_step):
    train_x, test_x, train_y, test_y = load_data()

    # Size of Layers
    x_size = train_x.shape[1]  # Input nodes: 23 features and 1 bias
    y_size = train_y.shape[1]  # Outcomes (2 types of party)

    # variables
    X = tf.placeholder("float", shape=[None, x_size])
    y = tf.placeholder("float", shape=[None, y_size])
    weights_1 = initialize_weights((x_size, h_size), stddev)
    weights_2 = initialize_weights((h_size, y_size), stddev)

    #compute forword
    y_pred = forward_propagation(X, weights_1, weights_2)
    #what get the largest outcome
    predict = tf.argmax(y_pred, dimension=1)
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_pred))
    updates_sgd = tf.train.GradientDescentOptimizer(sgd_step).minimize(cost)
    sess = tf.Session()
    # init = tf.global_variables_initializer()
    init = tf.initialize_all_variables()
    steps = 30
    sess.run(init)
    x  = np.arange(steps)
    test_acc = []
    train_acc = []
    print("Step, train accuracy, test accuracy")
    for step in range(steps):
        # Train with each example
        for i in range(len(train_x)):
            sess.run(updates_sgd, feed_dict={X: train_x[i: i + 1], y: train_y[i: i + 1]})

        train_accuracy = np.mean(np.argmax(train_y, axis=1) ==
                                 sess.run(predict, feed_dict={X: train_x, y: train_y}))
        test_accuracy = np.mean(np.argmax(test_y, axis=1) ==
                                sess.run(predict, feed_dict={X: test_x, y: test_y}))

        print("%d, %.2f%%, %.2f%%"
              % (step + 1, 100. * train_accuracy, 100. * test_accuracy))
        #x.append(step)
        test_acc.append(100. * test_accuracy)
        train_acc.append(100. * train_accuracy)

    t = [np.array(test_acc)]
    t.append(train_acc)
    title = "Steps vs Accuracy-No of hidden nodes: " + str(h_size) + ", sgd step:" + str(sgd_step) +             ", std dev:" + str(stddev)
    label = ['Test Accuracy', 'Train Accuracy']
    sess.close()
    import matplotlib.pyplot as plt
    plt.style.use('seaborn-whitegrid')
    
    plt.plot(x, train_acc, '-ok', color='red');

    plt.style.use('seaborn-whitegrid')
    
    plt.plot(x, test_acc, '-ok', color='green');
    plt.xlabel("step")
    plt.ylabel("Accuracy");
    max1.append(max(test_acc))

def main():
    run(128,0.1,0.01)


if __name__ == '__main__':
    main()


# red line=train accuracy
# green line=test accuracy
# 

# # model 2-smoller  step in gradient direction

# In[5]:


def load_data():

    data=input_data.iloc[:200, 3:21] # all rows, all the features and no labels
    target = input_data.iloc[:200, 22]  # all rows, label only

    # Prepend the column of 1s for bias
    L, W  = data.shape
    all_X = np.ones((L, W + 1))
    all_X[:, 1:] = data
    num_labels = len(np.unique(target))
    all_y = np.eye(num_labels)[target]
    return train_test_split(all_X, all_y, test_size=0.33, random_state=RANDOMSEED)




def initialize_weights(shape, stddev):
    weights = tf.random.normal(shape, stddev=stddev)
    return tf.Variable(weights)

def forward_propagation(X, weights_1, weights_2):
    sigmoid = tf.nn.sigmoid(tf.matmul(X, weights_1))
    y = tf.matmul(sigmoid, weights_2)
    return y

def run(h_size, stddev, sgd_step):
    train_x, test_x, train_y, test_y = load_iris_data()

    # Size of Layers
    x_size = train_x.shape[1]  # Input nodes: 23 features and 1 bias
    y_size = train_y.shape[1]  # Outcomes (2 types of party)

    # variables
    X = tf.placeholder("float", shape=[None, x_size])
    y = tf.placeholder("float", shape=[None, y_size])
    weights_1 = initialize_weights((x_size, h_size), stddev)
    weights_2 = initialize_weights((h_size, y_size), stddev)

    #compute forword
    y_pred = forward_propagation(X, weights_1, weights_2)
    #what get the largest outcome
    predict = tf.argmax(y_pred, dimension=1)
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_pred))
    updates_sgd = tf.train.GradientDescentOptimizer(sgd_step).minimize(cost)
    sess = tf.Session()
    # init = tf.global_variables_initializer()
    init = tf.initialize_all_variables()
    steps = 30
    sess.run(init)
    x  = np.arange(steps)
    test_acc = []
    train_acc = []
    print("Step, train accuracy, test accuracy")
    for step in range(steps):
        # Train with each example
        for i in range(len(train_x)):
            sess.run(updates_sgd, feed_dict={X: train_x[i: i + 1], y: train_y[i: i + 1]})

        train_accuracy = np.mean(np.argmax(train_y, axis=1) ==
                                 sess.run(predict, feed_dict={X: train_x, y: train_y}))
        test_accuracy = np.mean(np.argmax(test_y, axis=1) ==
                                sess.run(predict, feed_dict={X: test_x, y: test_y}))

        print("%d, %.2f%%, %.2f%%"
              % (step + 1, 100. * train_accuracy, 100. * test_accuracy))
        #x.append(step)
        test_acc.append(100. * test_accuracy)
        train_acc.append(100. * train_accuracy)

    t = [np.array(test_acc)]
    t.append(train_acc)
    title = "Steps vs Accuracy-No of hidden nodes: " + str(h_size) + ", sgd step:" + str(sgd_step) +             ", std dev:" + str(stddev)
    label = ['Test Accuracy', 'Train Accuracy']
    sess.close()
    import matplotlib.pyplot as plt
    plt.style.use('seaborn-whitegrid')
    
    plt.plot(x, train_acc, '-ok', color='red');

    plt.style.use('seaborn-whitegrid')
    
    plt.plot(x, test_acc, '-ok', color='green');
    plt.xlabel("step")
    plt.ylabel("Accuracy");
    max1.append(max(test_acc))

def main():
    run(128,0.1,0.001)


if __name__ == '__main__':
    main()


# red line=train accuracy
# green line=test accuracy
# 

# #model 3-overfitting

# In[6]:


def load_data():

    data=input_data.iloc[:200, 3:21] # all rows, all the features and no labels
    target = input_data.iloc[:200, 22]  # all rows, label only

    # Prepend the column of 1s for bias
    L, W  = data.shape
    all_X = np.ones((L, W + 1))
    all_X[:, 1:] = data
    num_labels = len(np.unique(target))
    all_y = np.eye(num_labels)[target]
    return train_test_split(all_X, all_y, test_size=0.33, random_state=RANDOMSEED)




def initialize_weights(shape, stddev):
    weights = tf.random.normal(shape, stddev=stddev)
    return tf.Variable(weights)

def forward_propagation(X, weights_1, weights_2):
    sigmoid = tf.nn.sigmoid(tf.matmul(X, weights_1))
    y = tf.matmul(sigmoid, weights_2)
    return y

def run(h_size, stddev, sgd_step):
    train_x, test_x, train_y, test_y = load_data()

    # Size of Layers
    x_size = train_x.shape[1]  # Input nodes: 23 features and 1 bias
    y_size = train_y.shape[1]  # Outcomes (2 types of party)

    # variables
    X = tf.placeholder("float", shape=[None, x_size])
    y = tf.placeholder("float", shape=[None, y_size])
    weights_1 = initialize_weights((x_size, h_size), stddev)
    weights_2 = initialize_weights((h_size, y_size), stddev)

    #compute forword
    y_pred = forward_propagation(X, weights_1, weights_2)
    #what get the largest outcome
    predict = tf.argmax(y_pred, dimension=1)
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_pred))
    updates_sgd = tf.train.GradientDescentOptimizer(sgd_step).minimize(cost)
    sess = tf.Session()
    # init = tf.global_variables_initializer()
    init = tf.initialize_all_variables()
    steps = 30
    sess.run(init)
    x  = np.arange(steps)
    test_acc = []
    train_acc = []
    print("Step, train accuracy, test accuracy")
    for step in range(steps):
        # Train with each example
        for i in range(len(train_x)):
            sess.run(updates_sgd, feed_dict={X: train_x[i: i + 1], y: train_y[i: i + 1]})

        train_accuracy = np.mean(np.argmax(train_y, axis=1) ==
                                 sess.run(predict, feed_dict={X: train_x, y: train_y}))
        test_accuracy = np.mean(np.argmax(test_y, axis=1) ==
                                sess.run(predict, feed_dict={X: test_x, y: test_y}))
        
        print("%d, %.2f%%, %.2f%%"
              % (step + 1, 100. * train_accuracy, 100. * test_accuracy))
        #x.append(step)
        test_acc.append(100. * test_accuracy)
        train_acc.append(100. * train_accuracy)

    t = [np.array(test_acc)]
    t.append(train_acc)
    title = "Steps vs Accuracy-No of hidden nodes: " + str(h_size) + ", sgd step:" + str(sgd_step) +             ", std dev:" + str(stddev)
    label = ['Test Accuracy', 'Train Accuracy']
    sess.close()
    import matplotlib.pyplot as plt
    plt.style.use('seaborn-whitegrid')
    
    plt.plot(x, train_acc, '-ok', color='red');

    plt.style.use('seaborn-whitegrid')
    
    plt.plot(x, test_acc, '-ok', color='green');
    plt.xlabel("step")
    plt.ylabel("Accuracy");
    max1.append(max(test_acc))

def main():
    run(140,0.1,0.001)


if __name__ == '__main__':
    main()


# red line=train accuracy
# green line=test accuracy
# 

# # model 4-under fitting

# In[8]:


def load_iris_data():

    data=input_data.iloc[:200, 3:21] # all rows, all the features and no labels
    target = input_data.iloc[:200, 22]  # all rows, label only

    # Prepend the column of 1s for bias
    L, W  = data.shape
    all_X = np.ones((L, W + 1))
    all_X[:, 1:] = data
    num_labels = len(np.unique(target))
    all_y = np.eye(num_labels)[target]
    return train_test_split(all_X, all_y, test_size=0.33, random_state=RANDOMSEED)




def initialize_weights(shape, stddev):
    weights = tf.random.normal(shape, stddev=stddev)
    return tf.Variable(weights)

def forward_propagation(X, weights_1, weights_2):
    sigmoid = tf.nn.sigmoid(tf.matmul(X, weights_1))
    y = tf.matmul(sigmoid, weights_2)
    return y

def run(h_size, stddev, sgd_step):
    train_x, test_x, train_y, test_y = load_iris_data()

    # Size of Layers
    x_size = train_x.shape[1]  # Input nodes: 23 features and 1 bias
    y_size = train_y.shape[1]  # Outcomes (2 types of party)

    # variables
    X = tf.placeholder("float", shape=[None, x_size])
    y = tf.placeholder("float", shape=[None, y_size])
    weights_1 = initialize_weights((x_size, h_size), stddev)
    weights_2 = initialize_weights((h_size, y_size), stddev)

    #compute forword
    y_pred = forward_propagation(X, weights_1, weights_2)
    #what get the largest outcome
    predict = tf.argmax(y_pred, dimension=1)
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_pred))
    updates_sgd = tf.train.GradientDescentOptimizer(sgd_step).minimize(cost)
    sess = tf.Session()
    # init = tf.global_variables_initializer()
    init = tf.initialize_all_variables()
    steps = 30
    sess.run(init)
    x  = np.arange(steps)
    test_acc = []
    train_acc = []
    print("Step, train accuracy, test accuracy")
    for step in range(steps):
        # Train with each example
        for i in range(len(train_x)):
            sess.run(updates_sgd, feed_dict={X: train_x[i: i + 1], y: train_y[i: i + 1]})

        train_accuracy = np.mean(np.argmax(train_y, axis=1) ==
                                 sess.run(predict, feed_dict={X: train_x, y: train_y}))
        test_accuracy = np.mean(np.argmax(test_y, axis=1) ==
                                sess.run(predict, feed_dict={X: test_x, y: test_y}))
        
        print("%d, %.2f%%, %.2f%%"
              % (step + 1, 100. * train_accuracy, 100. * test_accuracy))
        #x.append(step)
        test_acc.append(100. * test_accuracy)
        train_acc.append(100. * train_accuracy)

    t = [np.array(test_acc)]
    t.append(train_acc)
    title = "Steps vs Accuracy-No of hidden nodes: " + str(h_size) + ", sgd step:" + str(sgd_step) +             ", std dev:" + str(stddev)
    label = ['Test Accuracy', 'Train Accuracy']
    sess.close()
    import matplotlib.pyplot as plt
    plt.style.use('seaborn-whitegrid')
    
    plt.plot(x, train_acc, '-ok', color='red');

    plt.style.use('seaborn-whitegrid')
    
    plt.plot(x, test_acc, '-ok', color='green');
    plt.xlabel("step")
    plt.ylabel("Accuracy");
    
    max1.append(max(test_acc))

def main():
    run(80,0.1,0.001)


if __name__ == '__main__':
    main()


# In[9]:



import matplotlib.pyplot as plt
xx=[1,2,3,4]
plt.style.use('seaborn-whitegrid')
    
plt.plot(xx, max1, '-ok', color='blue');
plt.xlabel("model number")
plt.ylabel("max Accuracy");


# In[ ]:




