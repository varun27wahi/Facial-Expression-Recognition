function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));


m = size(X, 1);
         
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

y_matrix = eye(num_labels)(y,:);

a1 = [ones(m, 1) X];
z2 = a1 * Theta1';
a2 = sigmoid(z2);
a2 = [ones(m, 1), a2];
z3 = a2* Theta2';
a3 = sigmoid(z3);

term1 = -y_matrix .* log(a3);
term2 = (1 .- y_matrix) .* log(1 .- a3);
JUnReg = (1/m) * (sum(sum(term1 - term2)));

Theta1_temp = Theta1(:, 2:end);
Theta2_temp = Theta2(:, 2:end);

sumOfSquares1 = sum(sum(Theta1_temp .^ 2));
sumOfSquares2 = sum(sum(Theta2_temp .^ 2));
regularizationTerm = (lambda / (2*m)) * (sumOfSquares1 + sumOfSquares2);

J = JUnReg + regularizationTerm;

d3 = a3 - y_matrix;
d2 = (d3 * Theta2(:,2:end)) .* sigmoidGradient(z2);

Delta1 = d2' * a1;
Delta2 = d3' * a2;

Theta1(:, 1) = 0;
Theta2(:, 1) = 0;

Theta1_grad = ((1 / m) * Delta1) + ((lambda / m) * Theta1);
Theta2_grad = ((1 / m) * Delta2) + ((lambda / m) * Theta2);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
