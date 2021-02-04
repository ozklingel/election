#!/usr/bin/env python
# coding: utf-8

# In[6]:



import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
import h5py

from keras.models import Sequential
from keras.layers import Conv2D, Dense, Flatten, Dropout, MaxPool2D
from keras.optimizers import Adam, RMSprop
from keras.initializers import RandomNormal
import matplotlib.pyplot as plt
#import seaborn as sns
import os
#print(os.listdir("../input"))
trainFile=h5py.File('C:/Users/97254/Downloads/train_happy.h5', "r")
#trainFile = h5py.File('../input/train_happy.h5')
testFile=h5py.File('C:/Users/97254/Downloads/test_happy.h5', "r")
#testFile = h5py.File('../test_happy.h5')


# In[7]:


train_x = np.array(trainFile['train_set_x'][:])
train_y = np.array(trainFile['train_set_y'][:])

test_x = np.array(testFile['test_set_x'][:])
test_y = np.array(testFile['test_set_y'][:])
print(train_x.shape)
print(train_y.shape)


# In[8]:


train_y = train_y.reshape((1, train_y.shape[0]))
test_y = test_y.reshape((1, test_y.shape[0]))

print(train_y.shape)
print(test_y.shape)


# In[9]:


plt.imshow(train_x[0])


# In[12]:


X_train = train_x / 255.0
X_test = test_x / 255.0

y_train = train_y.T
y_test = test_y.T


# In[13]:


model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3,3), activation='relu', padding='Same', input_shape=(64,64,3)))
model.add(Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='Same'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='Same'))

model.add(MaxPool2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])


# In[14]:


epochs = 10
batch_size = 30
history = model.fit(x=X_train, y=y_train, epochs=epochs, verbose=2,batch_size=batch_size)


# In[15]:


test_score = model.evaluate(X_test, y_test, verbose=1)


# In[16]:


print('test loss:', test_score[0])
print('test accuracy:', test_score[1])


# In[20]:


print(history.history.keys())


# In[21]:


training_accuracy = history.history['accuracy']
training_loss = history.history['loss']

E = range(len(training_accuracy))
plt.plot(E, training_accuracy, color='red', label='Training accuracy')
plt.title('epochs vs Training accuracy')
plt.legend()

plt.figure()
plt.plot(E, training_loss, color='red', label='Training Loss')
plt.title('epochs vs Training Loss')
plt.legend()


# In[ ]:




