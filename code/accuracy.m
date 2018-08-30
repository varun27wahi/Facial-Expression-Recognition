load('../data/data_train.mat');
pixels = double(pixels);

load('../data/data_test.mat');
pixels_test = double(pixels_test);

load('../data/final_weights.mat');

pixels_norm = featureNormalize(pixels);
pixels_norm_test = featureNormalize(pixels_test);

% Accuracy
pred = predict(Theta1, Theta2, pixels_norm);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == emotion)) * 100);

pred_test = predict(Theta1, Theta2, pixels_norm_test);

fprintf('\nTesting Set Accuracy: %f\n', mean(double(pred_test == emotion_test)) * 100);
