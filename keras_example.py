from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
import numpy

numpy.random.seed(7)

#Load data
#===========================================================================
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
X = dataset[:,0:8]
Y = dataset[:,8]

#Define model
#Create CNN by adding one layer at a time, defining the number of nodes and its activation function
#   Exception:  First layer should have input_dim be the number of inputs from the dataset.
#               Inputs is the number of features you are measuring. 
#===========================================================================
model = Sequential() #Choose the type of model
model.add(Dense(12, input_dim=8, activation='relu')) #12 nodes with 8 inputs
model.add(Dense(8, activation='relu')) #8 nodes
model.add(Dense(1, activation='sigmoid')) #Output

#Compile
#Define the loss function, optimizer
#List of loss function and optimizer is available on the keras website
#===========================================================================

#Adam optimizer
#-------------------------------------------
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) #77%-78% accuracy

#SGD - Stochastic Gradient Descent
#-------------------------------------------
#sgd = optimizers.SGD(lr=0.01, clipnorm=1)
#model.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy']) #65% accuracy

#Trains model
#Define epochs and batch size
#===========================================================================
model.fit(X, Y, epochs=150, batch_size=10)

#Evaluate model
#===========================================================================
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
