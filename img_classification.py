import tensorflow as tf
from tensorflow.keras.datasets import mnist
from matplotlib import pyplot as plt
from tensorflow.keras.utils import to_categorical
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense



tf.logging.set_verbosity(tf.logging.ERROR)
#print("tensorflow version",tf.__version__)

(x_train,y_train),(x_test,y_test)=mnist.load_data()
#print("x_train shape",x_train.shape)
#print("y_train shape",y_train.shape)
#print("x_test shape",x_test.shape)
#print("y_test shape",y_test.shape)

#plt.imshow(x_train[0],cmap="binary")
#plt.show()

#for label

#print(y_train[0])

#for printing all the classes in mnist

#print(set(y_train))

#one hot encoding
#1=[0,1,0,0,0,0,0,0,0,0]#set according to index
#2=[0,0,1,0,0,0,0,0,0,0]

y_train_encoded=to_categorical(y_train)
y_test_encoded=to_categorical(y_test)

#print(y_train_encoded[0])


#print("y_train_encoded shape:",y_train_encoded.shape)
#print("y_test_encoded shape:",y_test_encoded.shape)


#print(y_train_encoded[0])

#here all the pixels of x images are features(28*28) and they are multiplied with weights.
#128 nodes for one hidden layer and 10 nodes as output.
#input will have 784 features(28*28)

#reshaping x features
x_train_reshape=np.reshape(x_train,(60000,784))
x_test_reshape=np.reshape(x_test,(10000,784))
#print("x_train_reshape shape:",x_train_reshape.shape)
#print("x_test_reshape shape:",x_test_reshape.shape)


#print(set(x_train_reshape[0]))

#data normalizing

x_mean=np.mean(x_train_reshape)
x_std=np.std(x_train_reshape)

epsilon=1e-10

x_train_norm=(x_train_reshape-x_mean)/(x_std+epsilon)
x_test_norm=(x_test_reshape-x_mean)/(x_std+epsilon)

#print(set(x_train_norm[0]))


model=Sequential([
	Dense(128,activation='relu',input_shape=(784,)),
	Dense(128,activation='relu'),
	Dense(10,activation='softmax')
])



model.compile(
	optimizer='sgd',
	loss='categorical_crossentropy',
	metrics=['accuracy']
	)
model.summary()

model.fit(x_train_norm,y_train_encoded,epochs=3)

_,accuracy=model.evaluate(x_test_norm,y_test_encoded)
print("test accuracy :",accuracy*100)

preds=model.predict(x_test_norm)
print("preds shape:",preds.shape)

'''plt.figure(figsize=(12,12))
start_index=0
for i in range(30):
	plt.subplot(10,10,i+1)
	plt.grid(False)
	plt.xticks([])
	plt.yticks([])

	pred=np.argmax(preds[start_index])
	ground_truth=y_test[start_index+i]

	col='g'
	if pred !=ground_truth:
		col='r'


	plt.xlabel("i={},pred={},ground_truth={}".format(start_index+i,pred,ground_truth),color=col)
	plt.imshow(x_test[start_index],cmap='binary')
plt.show()'''


plt.figure(figsize = (12, 12))
start_index = 0
for i in range(25):
	plt.subplot(5, 5, i + 1)
	plt.grid(False)
	plt.xticks([])
	plt.yticks([])

	pred = np.argmax(preds[start_index + i])
	actual = np.argmax(y_test_encoded[start_index + i])
	col = 'g'
	if pred != actual:
		col = 'r'
	plt.xlabel('i={} | pred={} | true={}'.format(start_index + i, pred, actual), color = col)
	plt.imshow(x_test[start_index + i], cmap='binary')
plt.show()
   


plt.plot(preds[8])
plt.show()

