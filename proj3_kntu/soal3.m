% Load dataset
dataset = readtable('ballbeam.dat');

input_data = dataset{:, 1}; % Input data
output_data = dataset{:, 2}; % Output data

% Normalize data
input_data = (input_data - min(input_data)) / (max(input_data) - min(input_data));
output_data = (output_data - min(output_data)) / (max(output_data) - min(output_data));

% Split data into training and testing sets
train_ratio = 0.8; % 80% for training
num_train_samples = floor(train_ratio * length(input_data));

% Shuffle data
rng(42); % Set seed for reproducibility
shuffled_indices = randperm(length(input_data));
train_indices = shuffled_indices(1:num_train_samples);
test_indices = shuffled_indices(num_train_samples+1:end);

% Create training and testing sets
train_input = input_data(train_indices);
train_output = output_data(train_indices);
test_input = input_data(test_indices);
test_output = output_data(test_indices);

% Configure ANFIS options
fis_options = genfisOptions('GridPartition');
fis_options.NumMembershipFunctions = 3;
fis_options.InputMembershipFunctionType = 'gbellmf';

% Generate initial FIS
initial_fis = genfis(train_input, train_output, fis_options);

% Train ANFIS
training_data = [train_input, train_output];
training_options = anfisOptions('InitialFis', initial_fis, 'EpochNumber', 100);
trained_network = anfis(training_data, training_options);

% Make predictions
train_predictions = evalfis(trained_network, train_input);
test_predictions = evalfis(trained_network, test_input);

% Calculate performance metrics
train_mse = mean((train_output - train_predictions).^2);
test_mse = mean((test_output - test_predictions).^2);

% Display performance metrics
fprintf('Training MSE: %.4f\n', train_mse);
fprintf('Testing MSE: %.4f\n', test_mse);

% Plot training results
figure;
hold on;
plot(train_output, 'Color', [0.2, 0.6, 0.8], 'LineWidth', 1.5, 'DisplayName', 'Actual Training Data'); % آبی روشن
plot(train_predictions, 'Color', [0.9, 0.4, 0.1], 'LineWidth', 1.5, 'LineStyle', '--', 'DisplayName', 'Predicted Training Data'); % نارنجی
xlabel('Sample Index');
ylabel('Output');
title('Training Data vs Predictions');
legend;
grid on;
hold off;

% Plot testing results
figure;
hold on;
plot(test_output, 'Color', [0.1, 0.7, 0.3], 'LineWidth', 1.5, 'DisplayName', 'Actual Testing Data'); % سبز
plot(test_predictions, 'Color', [0.8, 0.2, 0.6], 'LineWidth', 1.5, 'LineStyle', '--', 'DisplayName', 'Predicted Testing Data'); % صورتی
xlabel('Sample Index');
ylabel('Output');
title('Testing Data vs Predictions');
legend;
grid on;
hold off;