function p = predict(Theta1, Theta2, X)
%   Bob LoGalbo's prediction function of a trained neural network.
%   Predict the label of an input given a trained neural network
%   p = predict(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)
%
%   Calculate the number of images to be predicted i.e.
%   the number of rows
%
      m = size(X, 1);
%
%   This calculates the sigmoid of the input values, with the bias term
%   i.e. the activation function or hypothesis of the first layer.  
%
      h1 = 1.0 ./ (1.0 + exp(-[ones(m, 1) X] * Theta1'));
%
%   This calculates the sigmoid of the input values, to the hidden layer
%   with the bias term i.e. the activation function or hypothesis 
%   of the hidden layer.  
%   
%   The output of the hidden layer makes a prediction i.e.
%   the label with highest value is the most probable and is 
%   selected.
%   
      h2 = 1.0 ./ (1.0 + exp(-[ones(m, 1) h1] * Theta2'));
      [dummy, p] = max(h2, [], 2);
% =========================================================================


end
