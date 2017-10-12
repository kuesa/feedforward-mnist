import numpy as np 
import random
import math
from mnist import MNIST
import cv2

#def sigmoid(x, derivative=False):
#    if (derivative == True):
#        for i in range(0, len(x)):
#            for k in range(len(x[i])):
#                if x[i][k] > 0:
#                    x[i][k] = 1
#                else:
#                    x[i][k] = 0
#        return x
#    for i in range(0, len(x)):
#        for k in range(0, len(x[i])):
#            if x[i][k] > 0:
#                pass  # do nothing since it would be effectively replacing x with x
#            else:
#                x[i][k] = 0
#    return x

def sigmoid(x, derivative=False):
    if (derivative == True):
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))

mndata = MNIST('D:/Programming/ML/MNIST Network 2017/data/')
images, labels = mndata.load_training()

np.random.seed(1)

syn0 = 2*np.random.random((784, 16))-1
syn1 = 2*np.random.random((16,10))-1

alpha = 0.01

right = 0
wrong = 0

#forward propagation
for j in range(100000):
	dex = random.randint(0, 59999)
	X = np.atleast_2d(np.asarray(images[dex]) / 255)
	y = labels[dex]

	l0 = X
	l1 = sigmoid(np.dot(l0, syn0))
	l2 = sigmoid(np.dot(l1, syn1))

	guess = np.argmax(l2)
	l2_error = l2 
	l2_error[0, guess] = 1-l2_error[0, guess]
	total_error = np.sum(l2_error)

	#archaic error calculation
	#for i in range(10):
	#	if(np.argmax(l2) == i): 
	#		total_error += (1-l2[i])
	#		l2_error[i] = (1-l2[i])
	#	else: 
	#		total_error += l2[i]
	#		l2_error[i] = l2[i]

	#delta = error*deriv
	l2_delta = l2_error * sigmoid(l2, True)
	l1_error = l2_delta.dot(syn1.T)
	l1_delta = l1_error * sigmoid(l1, True)

	#l1 & l2_delta
	syn1 -= (alpha * np.dot(l1.T, l2_delta))
	syn0 -= (alpha * np.dot(l0.T, l1_delta))

	#display image - DO NOT ENABLE THIS!!!!!!
	#cv2.imshow('x', np.reshape(X, (28, 28)))
	#cv2.destroyAllWindows()

	if(guess == labels[dex]):
		correct = "Yes."
		right += 1
	else:
		correct = "No."
		wrong += 1

	#print("[Guess: " + str(guess) + "] [Answer: " + str(labels[dex]) + "] [Correct? : " + correct + "]")
	#print("[Correct Answers: " + str(right) + "] [Incorrect Answers: " + str(wrong) + "]")
	print(total_error)

percentRight = (right / (right + wrong)) * 100
print("Correct Answers: " + str(right))
print("Incorrect Answers: " + str(wrong))
print("Percentage of Correct Answers: " + str(percentRight) + "%")

