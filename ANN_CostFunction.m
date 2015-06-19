%  Bob LoGalbo's Artificial Neural Network (ANN)
%  regularized cost function and gradient descent algorithm.
%  
%  ANN_CostFunction implements the neural network cost function 
%  and gradient calculation for a two layer
%  neural network which performs one-vs-all classification.  
%  This neural network is regularized to prevent overfitting.
%  This module is completely vectorized without loops for fast
%  linear algebra implementation.
%
function [J gradients] = ANN_CostFunction(ANN_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%
%
%   [J gradients] = ANN_CostFunction(ANN_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost (J) and gradient (gradients) of the neural network. 
%   The gradients are the partial derivatives of the cost J w.r.t. a
%   particular weighting parameter.  The cost function and gradients 
%   can then be fed to fmincg which shall iterate to find where
%   the partial derivatives are zero to find a minima w.r.t. J
%   i.e. the parameters which minimize the cost of predictive error.
%   
%   Theta1 are the parameter weights which get multiplied times
%   the input values and Theta2 are the hidden
%   layer parameter values which get multiplied times
%   the hidden layer activation function outputs.
%   
%   The parameters (theta) for the 
%   neural network are "unrolled" from the vector
%   ANN_params.
%
      Theta1 = reshape(ANN_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

      Theta2 = reshape(ANN_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));
%
%    Each row of input X contains all of the fixed point values of
%    one image which contains one handwritten symbol (label). 
% 
%    The number of images m (i.e. number of rows):
%
       m = size(X, 1);
%
%     Adding bias terms to the input matrix for forward propagation
%
        X = [ones(m, 1) X];
%
%     Use a sigmoid as the input neural activation function
% 
%     Please note:  sigmoid(z) = 1.0 ./ (1.0 + exp(-z));
%
        a2=sigmoid(X*Theta1');
%
%     Adding bias terms to hidden layer for forward propagation
%       
        a2=[ones(m, 1) a2];
%
%     Use a sigmoid as the hidden layer neural activation function
%
         h = sigmoid(a2*Theta2');
%
%     Remove bias for both input layer 
%     and hidden layer eliminating regularization of bias
%     parameter weightings when computing the cost:
%
        thetaUR1=Theta1;
        thetaUR1(:,1) = 0;
        thetaUR2=Theta2;
        thetaUR2(:,1) = 0;
%
%     The vector y (ground truth) passed into the function is a vector of labels
%     containing values from 1..K. This vector is mapped into a 
%     binary vector of 1's and 0's to be used with the neural network
%     cost function (described in the next paragraph).
%
%     The following code translates the column vector of ground truth (y).  
%     Each element of the vector is an integer representing 
%     the ID of the training image.  The following code tranlates 
%     the ground truth vector to a matrix where each row has a binary
%     indicator (i.e. the value 1) in the row's position representing
%     the label in the ground truth vector.  For example, if the digits 
%     0 - 9 were to be classified and y(i) = 4, the ith row of
%     y1VsAll = [0 0 0 0 1 0 0 0 0 0].  If the jth row of y = 8, the
%     jth row of y1VsAll = [0 0 0 0 0 0 0 0 1 0].
%
        y1VsAll = (repmat(1:num_labels,size(y,1),1) ==  repmat(y,1,num_labels) );
%
%     The following implements the cost of training error  including 
%     the intentional penalty incurred by regularization.
%     The cost is a function of penalizing the Bayesian, 
%     conditional error i.e. p(0|1) & p(1|0) and "rewarding" 
%     Bayesian success i.e. p(0|0) and p(1|1).
% 
        J = sum((1/m)*( ( diag ( -y1VsAll' *log(h) ) )  - (diag ( (1-y1VsAll')*(log(1-h) ) ) ) ) ) + ( (lambda/(2*m) ) *  ( sum (diag (thetaUR1' * thetaUR1) )+ sum (diag (thetaUR2' * thetaUR2) ) ) );
%
%    The backpropagation algorithm computes the gradients
%    Theta1_grad and Theta2_grad.   These are the partial derivatives of
%    the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%    Theta2_grad, respectively.  There are several steps before the final 
%    calculation of Theta1_grad & Theta2_grad.
%
%    del3 is the error between the ground truth and hypothesis
%    i.e. the final activation functions of the neural network
%    executed by the hidden layer.
%     
       del3 = h - y1VsAll;
% 
%    Note that the error above is weighted times the parameters moving
%    backward through the network (i.e. the hidden layer parameters)
%    then times the input layer's activation function output with the
%    bias terms included. 
%
%    The input layer's activation function 
%    makes a decision based upon the input values, 
%    including the bias term.  The output of the input layer's decision
%    is then weighted against the first layer of parameters.  
%    The value del2 now includes the total error i.e. both 
%    the forward propagation to the hidden layer and the
%    backward error propagation to the hidden layer.
% 
%    Note that the sigmoidGradient function below is:
%    sigmoid(z).*(1-sigmoid(z)) i.e. 
%    the sigmoidGradient function is the derivative of the sigmoid
%    function w.r.t. z.
%
       del2 = (del3*Theta2).*sigmoidGradient([ones(m,1) (X*Theta1')]);
%
%     Now drop the value in the bias column of the total error when calculating 
%     the partial derivative of the cost function wrt each 
%     parameter i.e. the gradient of the cost at the particular parameter.
%     The bias terms are constants and the partial derivative of a constant = 0.
%
       del2 = del2(:,2:end);
%
%    The following calculates the regularized parameters.
%
       Theta1Reg = [zeros([size(Theta1,1),1]) ((lambda/m)*Theta1(:,2:end))];   
       Theta2Reg = [zeros([size(Theta2,1),1]) ((lambda/m)*Theta2(:,2:end))];      
%
%    The following calculates the final value of the partial derivative
%    of the cost w.r.t. the parameters i.e. the gradients.  Given there
%    are two sets of parameters (i.e. from the input values to the
%    hidden layer and from the hidden layer to the output), two sets of
%    gradients are calculated.
% 
%    The following is simply the partial derivative of the cost function 
%    w.r.t. the parameters.
%  
       Theta2_grad = (1/m) * (del3' * a2) + Theta2Reg;
       Theta1_grad = (1/m) * (del2' * X) + Theta1Reg;
%
%     Gradients must be assembled into 1 matrix to be fed into the
%     minimization function.
%
        gradients = [Theta1_grad(:) ; Theta2_grad(:)];

end
