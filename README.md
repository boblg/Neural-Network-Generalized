# Generalized-Neural-Network
Neural network w/ one hidden layer generalized to any size input, number of hidden layer nodes or logistical label classifiers.

Bob LoGalbo ver 1.0

ANN_costfunction.m is a publishable, non-proprietary, 
homespun, 1 hidden layer neural network illustrating
forward & backpropagation training relying on the partial derivative
gradients of the cost function w.r.t. the training parameters.  

Included are the backpropagation error calculations, 
cost function, the partial derivatives of the 
cost function, the activation functions, the partial derivatives of the
activation functions, parameter regularization and bias terms.

The function returns the cost and parameter gradient matrices.

The function is generalized to any size input matrix, output matrix or 
hidden layer and is completely linearized without loops.

Activation functions other than a sigmoid can easily replace 
those coded.

The function is coded in Matlab/Octave.

A second file (predict.m) is also included which is the elementary prediction function 
which results from this 1 hidden layer network post parameter training.  