% Initialization
clear ; close all; clc

% Layer sizes
input_layer_size  = 2304;  
hidden_layer_size = 100;   
num_labels = 7;

% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

load('../data/data_try.mat');
m = size(pixels, 1);
pixels = double(pixels);

% Test data
load('../data/data_try2.mat');
m_try2 = size(pixels_try2, 1);
pixels_try2 = double(pixels_try2);

load('../data/bat_image.mat');
Im = double(Im);

% Randomly select 100 data points to display
sel = randperm(size(pixels, 1));
sel = sel(1:100);

displayData(pixels(sel, :));

pixels_norm = featureNormalize(pixels);
pixels_norm_try2 = featureNormalize(pixels_try2);
Im_norm = featureNormalize(Im);

fprintf('Program paused. Press enter to continue.\n');
pause;

% Initializing parameters 
fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];

fprintf('\nTraining Neural Network... \n')

options = optimset('MaxIter', 50);

lambda = 1;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, pixels_norm, emotion, lambda);


[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

fprintf('Program paused. Press enter to continue.\n');
pause;

% Accuracy
[val pred] = predict(Theta1, Theta2, pixels_norm);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == emotion)) * 100);

% [val_test pred_test] = predict(Theta1, Theta2, pixels_norm_try2);

% fprintf('\nTesting Set Accuracy: %f\n', mean(double(pred_test == emotion_try2)) * 100);

fprintf('Program paused. Press enter to continue.\n');
pause;

fprintf('\nDisplaying Bat Image\n');
displayData(Im);

[maxValBat predBat] = predict(Theta1, Theta2, Im_norm);

predictedEmotion = '';

switch predBat
	case 1
		predictedEmotion = 'Disgust';
	case 2
		predictedEmotion = 'Fear';
	case 3
		predictedEmotion = 'Happy';
	case 4
		predictedEmotion = 'Sad';
	case 5
		predictedEmotion = 'Surprise';
	case 6
		predictedEmotion = 'Neutral';
	case 7	
		predictedEmotion = 'Angry';
end

if maxValBat < 0.4
	fprintf('\nNeural Network Prediction: None, as the confidence is less than 0.4 (%f)\n', maxValBat);
else
	fprintf('\nNeural Network Prediction: %s, with confidence %f\n', predictedEmotion, maxValBat);
end
fprintf('Program paused. Press enter to continue.\n');
pause;

rp = randperm(m);

for i = 1:m
    % Display 
    fprintf('\nDisplaying Example Image\n');
    displayData(pixels(rp(i), :));

    [maxVal pred] = predict(Theta1, Theta2, pixels_norm(rp(i),:));

    predictedEmotion = '';

    switch pred
    	case 1
    		predictedEmotion = 'Disgust';
    	case 2
    		predictedEmotion = 'Fear';
    	case 3
    		predictedEmotion = 'Happy';
    	case 4
    		predictedEmotion = 'Sad';
    	case 5
    		predictedEmotion = 'Surprise';
    	case 6
    		predictedEmotion = 'Neutral';
    	case 7	
    		predictedEmotion = 'Angry';
    end

    fprintf('\nNeural Network Prediction: %s, with confidence %f\n', predictedEmotion, maxVal);
    
    % Pause with quit option
    s = input('Paused - press enter to continue, q to exit:','s');
    if s == 'q'
      break
    end
end

% rp = randperm(m_try2);

% for i = 1:m_try2
%     % Display 
%     fprintf('\nDisplaying Example Image\n');
%     displayData(pixels_try2(rp(i), :));

%     pred = predict(Theta1, Theta2, pixels_norm_try2(rp(i),:));

%     predictedEmotion = '';
%     switch pred
%     	case 1
%     		predictedEmotion = 'Disgust';
%     	case 2
%     		predictedEmotion = 'Fear';
%     	case 3
%     		predictedEmotion = 'Happy';
%     	case 4
%     		predictedEmotion = 'Sad';
%     	case 5
%     		predictedEmotion = 'Surprise';
%     	case 6
%     		predictedEmotion = 'Neutral';
%     	case 7	
%     		predictedEmotion = 'Angry';
%     end

%     fprintf('\nNeural Network Prediction: %s\n', predictedEmotion);
    
%     % Pause with quit option
%     s = input('Paused - press enter to continue, q to exit:','s');
%     if s == 'q'
%       break
%     end
% end
