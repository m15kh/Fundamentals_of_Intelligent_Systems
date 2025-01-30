% Load dataset
data = readtable('AirQualityUCI.csv', 'Delimiter', ';');

% Select relevant columns
features = {'CO_GT_', 'C6H6_GT_', 'PT08_S2_NMHC_'};
target = 'PT08_S1_CO_';

X = data(:, features);
Y = data(:, target);

% Convert table to array
X = table2array(X);
Y = table2array(Y);

% Replace -200 with NaN
X(X == -200) = NaN;
Y(Y == -200) = NaN;

% Remove rows with NaN
validRows = ~any(isnan([X, Y]), 2);
X = X(validRows, :);
Y = Y(validRows, :);

% Normalize features and target
X = normalize(X);
Y = normalize(Y);

% Split dataset
cv = cvpartition(size(X,1), 'HoldOut', 0.4);
x_train = X(training(cv), :);
y_train = Y(training(cv), :);
x_temp = X(test(cv), :);
y_temp = Y(test(cv), :);

cv2 = cvpartition(size(x_temp,1), 'HoldOut', 0.5);
x_val = x_temp(training(cv2), :);
y_val = y_temp(training(cv2), :);
x_test = x_temp(test(cv2), :);
y_test = y_temp(test(cv2), :);

% Train RBF Model
numCenters = 5;
net = newrb(x_train', y_train', 0.0, 1.0, numCenters, 1);
y_pred_rbf = sim(net, x_test');
y_pred_rbf = y_pred_rbf';

% Evaluate RBF Model
mse_rbf = mean((y_test - y_pred_rbf).^2);
r2_rbf = 1 - sum((y_test - y_pred_rbf).^2) / sum((y_test - mean(y_test)).^2);

fprintf('RBF Model - MSE: %.4f, R-Squared: %.4f\n', mse_rbf, r2_rbf);

% Train ANFIS Model
fismat = genfis1([x_train, y_train], 2, 'gaussmf');
anfis_opt = anfisOptions('EpochNumber', 20);
anfisModel = anfis([x_train, y_train], fismat, anfis_opt);
y_pred_anfis = evalfis(x_test, anfisModel);

% Evaluate ANFIS Model
mse_anfis = mean((y_test - y_pred_anfis).^2);
r2_anfis = 1 - sum((y_test - y_pred_anfis).^2) / sum((y_test - mean(y_test)).^2);

fprintf('ANFIS Model - MSE: %.4f, R-Squared: %.4f\n', mse_anfis, r2_anfis);

% Compare models
if mse_rbf < mse_anfis
    fprintf('The RBF model performed better with a lower MSE.\n');
else
    fprintf('The ANFIS model performed better with a lower MSE.\n');
end

if r2_rbf > r2_anfis
    fprintf('The RBF model performed better with a higher R-Squared.\n');
else
    fprintf('The ANFIS model performed better with a higher R-Squared.\n');
end

% Plot results
figure;
hold on;
plot(y_test, 'b-o', 'DisplayName', 'Actual Data');
plot(y_pred_rbf, 'r--s', 'DisplayName', 'RBF Predictions');
plot(y_pred_anfis, 'g-.d', 'DisplayName', 'ANFIS Predictions');
hold off;
legend;
xlabel('Test Sample Index');
ylabel('Target Value (Denormalized)');
title('Comparison of Actual Data, RBF Predictions, and ANFIS Predictions');
grid on;
