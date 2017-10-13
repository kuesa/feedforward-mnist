import numpy as np 
import random
from mnist import MNIST 
import cv2

def sigmoid(x, derivative=False):
    if (derivative == True):
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))

def makeY(inp):
	oof = np.zeros((1,10))
	oof[0, inp] = 1
	return oof

#our lovely dataset
mndata = MNIST('data/')
images, labels = mndata.load_testing()

#load synapse values from training
syn0 = np.load('sv/syn0.npy')
syn1 = np.load('sv/syn1.npy')

#set output image size
dst = np.array((280,280))

#looping
for j in range(10):
	#define input and expected output
	dex = random.randint(0, 9999)
	X = np.atleast_2d(np.asarray(images[dex]) / 255)
	y = makeY(labels[dex])

	#forward propagation
	l0 = X
	l1 = sigmoid(np.dot(l0, syn0))
	l2 = sigmoid(np.dot(l1, syn1))

	guess = np.argmax(l2)

	#ui stuff
	print(guess)
	src = np.reshape(X, (28,28))
	dst = cv2.resize(src, (280,280))
	
	cv2.imshow('display', dst)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
