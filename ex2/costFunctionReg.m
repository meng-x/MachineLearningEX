function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples
n = length(theta);

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


costsum = 0;

for i = 1:m
  costsum = costsum + (y(i)*log(sigmoid(X(i,:) * theta)) + (1 - y(i)) * log(1 - sigmoid(X(i,:) * theta)));
  % h(x(i)) = sigmoid(X(i,:) * theta)
end

J = -1 / m * costsum + lambda / (2 * m) * ([0 ones(1, n - 1)] * theta.^2);



grad(1) = 1 / m * X(:,1)' * (sigmoid(X * theta) - y);
for j = 2:n
  grad(j) = 1 / m * X(:,j)' * (sigmoid(X * theta) - y) + lambda / m * theta(j);
  % partial derivative term :  1 / m * ( X' * (h(X) - y))
end



% =============================================================

end
