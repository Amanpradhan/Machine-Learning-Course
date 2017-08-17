function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.

%h = sigmoid(X*theta);
%cost = (1/m)*sum((-y.*log(h))-((1-y).*log(1-(h))));
%regCost = (lambda/(2*m))*norm(theta([2:end]))^2;
%grad = (1/m).*X'*(h-y);
%regGrad = (lambda/m).*theta;
%regGrad(1) = 0;
%J = cost+regCost;
%grad = grad+regGrad;






h = (X * theta);
cost = (1/(2*m) * (sum((h - y) .^ 2)));
regCost = (lambda/(2 * m)) * sum(theta(2:end).^2);

J = cost + regCost;

% Calculate linear regression gradient.
%putting theta(1) = 0 since it i always 0
theta(1) = 0;
grad = (1 / m) .* (X' * (h - y)) + ((lambda / m) .* theta);











% =========================================================================

grad = grad(:);

end
