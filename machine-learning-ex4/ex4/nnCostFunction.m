function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.


%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m


A1 = [ones([m, 1]) X];
A2 = sigmoid(A1 * Theta1');
A2 = [ones([m, 1])  A2];

for k = 1:1:num_labels 
    Theta2k = Theta2(k,:);
    J = J  -  ((y == k)' * log(sigmoid(A2 * Theta2k')) + (1 - (y == k)') * log(1 - sigmoid(A2 * Theta2k')) )/m;
end
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
%

%四层layer 最后一层输出层的delta是delta2
A3 = sigmoid(A2 * Theta2');

I = eye(num_labels);
Y = zeros(m, num_labels);
for i=1:m
  Y(i, :)= I(y(i), :);
end

Sigma3 = A3 - Y;

%四层layer 往前一层和a1相减的delta是delta1
Sigma2 =  (Sigma3 * Theta2) .* (A2 .* (1 - A2));

tmp1 = Theta1;
tmp1(:,1) = zeros(length(tmp1(:,1)), 1);
%第一行是相当于a1向a2计算的时候增加的（增加a0，所以西塔2会多一个参数）

tmp2 = Theta2;
tmp2(:,1) = zeros(length(tmp2(:,1)), 1);
J = J +  lambda * (sum(sum(tmp1 .* tmp1)) + sum(sum(tmp2 .* tmp2))) / (2 * m );

d2 = Sigma3' * A2;
d1 = Sigma2' * A1;
d1(1,:) = [];

%靠 这里PPT 好像有问题，原 PPT 里lambda是不除以m的
Theta1_grad = (d1)/m + (lambda/m) * tmp1;
Theta2_grad = (d2)/m + (lambda/m) * tmp2;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%




% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
