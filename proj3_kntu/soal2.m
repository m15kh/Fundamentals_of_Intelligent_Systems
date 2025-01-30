% Parameters
b = 2; % Distance between the front and rear axles (wheelbase)
num_steps = 1000; % Number of steps in the simulation
step_size = -0.1; % Step size for backward movement (negative for backward)

% Initial conditions
x = 5; % Initial x position (start away from target)
y = 10; % Initial y position
phi = pi/6; % Initial orientation (in radians)

% Define fuzzy inference system (FIS)
fis = mamfis('Name', 'CarParking');

% Input: distance_to_target
fis = addInput(fis, [-10 10], 'Name', 'distance_to_target');
fis = addMF(fis, 'distance_to_target', 'trapmf', [-10 -10 -1 1], 'Name', 'Negative');
fis = addMF(fis, 'distance_to_target', 'trapmf', [-1 1 10 10], 'Name', 'Positive');

% Input: phi (orientation)
fis = addInput(fis, [0 2*pi], 'Name', 'phi');
fis = addMF(fis, 'phi', 'trimf', [0 pi/2 pi+0.1], 'Name', 'Up'); % Looking Up
fis = addMF(fis, 'phi', 'trimf', [pi-0.1 3*pi/2 pi*2], 'Name', 'Down'); % Looking Down
fis = addMF(fis, 'phi', 'trimf', [3*pi/2-0.1 2*pi 2*pi], 'Name', 'Right'); % Looking Right
fis = addMF(fis, 'phi', 'trimf', [pi/2-0.1 pi pi], 'Name', 'Left'); % Looking Left

% Output: steering_angle
fis = addOutput(fis, [-pi/4 pi/4], 'Name', 'steering_angle');
fis = addMF(fis, 'steering_angle', 'trimf', [-pi/4 -pi/8 0], 'Name', 'Left');
fis = addMF(fis, 'steering_angle', 'trimf', [-pi/8 0 pi/8], 'Name', 'Center');
fis = addMF(fis, 'steering_angle', 'trimf', [0 pi/8 pi/4], 'Name', 'Right');

% Add fuzzy rules
ruleList = [ ...
    1 1 3 1 1; ... % If distance_to_target is Negative and phi is Up, steering_angle is Center
    2 1 1 1 1; ... % If distance_to_target is Positive and phi is Down, steering_angle is Left
    1 2 1 1 1; ... % If distance_to_target is Negative and phi is Right, steering_angle is Left
    2 2 3 1 1; ... % If distance_to_target is Positive and phi is Left, steering_angle is Right
    0 3 1 1 1; ... % If distance_to_target is Positive and phi is Up, steering_angle is Center
    0 4 3 1 1;
];

fis = addRule(fis, ruleList);

% Simulation loop with fuzzy logic
figure;
hold on;
grid on;
axis equal;
title('Car Parking Simulation (Fuzzy Logic)');
xlabel('X Position');
ylabel('Y Position');

for t = 1:num_steps
    % Calculate inputs to the FIS
    distance_to_target = 10 - x; % Target x = 10 (can be negative or positive)
    
    % Correct phi to be in the range [0, 2*pi]
    phi_corrected = 0;
    if phi > 0    
        phi_corrected = mod(phi, 2*pi);
    else
        phi_corrected = 2*pi - mod(abs(phi), 2*pi);
    end
     
    % Evaluate fuzzy logic
    theta = evalfis(fis, [distance_to_target, phi_corrected]); % Steering angle

    % Update the car's position and orientation using corrected formulas
    x = x + step_size * cos(phi);
    y = y + step_size * sin(phi);
    phi = phi + step_size * tan(theta) / b;

    % Plot the current position and orientation
    quiver(x, y, cos(phi), sin(phi), 0.5, 'r');
    plot(x, y, 'red');
    pause(0.001); % Pause to simulate real-time plotting
end

writeFIS(fis, 'CarParking.fis');

hold off;
