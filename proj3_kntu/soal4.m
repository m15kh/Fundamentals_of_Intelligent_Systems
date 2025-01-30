clear
close all

%% Initializing

M = 4; %Number of membership functions (Based on 1st step of fuzzy system design)
num_training = 200; % Number of training
total_num = 700;
landa = 0.1; % A constant stepsize

% Preallocation
x_bar = zeros(num_training, M);
g_bar = zeros(num_training, M);
sigma = zeros(num_training, M); 
y = zeros(total_num, 1);
u = zeros(total_num, 1);
x = zeros(total_num, 1);
y_hat = zeros(total_num, 1);
f_hat = zeros(total_num, 1);
z = zeros(total_num, 1);
g_u = zeros(total_num, 1);

u(1) = -1+2*rand;
y(1) = 0;
g_u(1) = 0.6*sin(pi*u(1))+0.3*sin(3*pi*u(1))+0.1*sin(5*pi*u(1));
f_hat(1) = g_u(1);


%% Based on the 1st step of fuzzy system design

u_min = -1;
u_max = 1;
h = (u_max-u_min)/(M-1);

for k = 1:M
    x_bar(1, k) = -1+h*(k-1);
    u(1,k) = x_bar(1, k);
    g_bar(1,k) = 0.6*sin(pi*u(1,k))+0.3*sin(3*pi*u(1,k))+0.1*sin(5*pi*u(1,k));
end

sigma(1,1:M) = (max(u(1,:))-min(u(1,:)))/M;

x_bar(2,:) = x_bar(1, :);
g_bar(2,: ) = g_bar(1, :);
sigma(2, :) = sigma(1,:);
x_bar_initial = x_bar(1, :);
sigma_initial = sigma(1, :);
y_bar_initial = g_bar(1,:);

%% Based on the 2nd and 3rd step of fuzzy system design for q=2: num_training

for q = 2:num_training
    b = 0;a = 0;
    x(q) = -1+2*rand;
    u(q) = x(q);
    g_u(q) = 0.6*sin(pi*u(q))+0.3*sin(3*pi*u(q))+0.1*sin(5*pi*u(q));
    
    for l = 1:M
        z(l) = exp(-((x(q)-x_bar(q,l))/sigma(q, l))^2);
        b = b+z(l);
        a = a+g_bar(q, l)*z(l);
    end

    f_hat(q) = a/b;
    y(q+1) = 0.3*y(q)+0.6*y(q-1)+g_u(q);
    y_hat(q+1) = 0.3*y(q)+0.6*y(q-1)+f_hat(q);

    for l = 1:M
        g_bar(q+1,l) = g_bar(q,l)-landa*(f_hat(q)-g_u(q))*z(l)/b;
        x_bar(q+1,l) = x_bar(q,l)-landa*((f_hat(q)-g_u(q))/b)*(g_bar(q,l)-f_hat(q))*z(l)*2*(x(q)-x_bar(q,l))/(sigma(q,l)^2);
        sigma (q+1,l) = sigma(q, l)-landa*((f_hat(q)-g_u(q))/b)*(g_bar(q,l)-f_hat(q))*z(l)*2*(x(l)-x_bar(q,l))^2/(sigma(q,l)^3);
    end
end

x_bar_final = x_bar(num_training,:);
sigma_final = sigma(num_training,:);
g_bar_final = g_bar(num_training,:);

for q = num_training:700
    b = 0;
    a = 0;
    x(q) = sin(2*q*pi/200);
    u(q) = x(q);

    g_u(q) = 0.6*sin(pi*u(q))+0.3*sin(3*pi*u(q))+0.1*sin(5*pi*u(q));
    
    for l = 1:M
        z(l) = exp(-((x(q)-x_bar(num_training,l))/sigma(num_training, l))^2);
        b = b+z(l);
        a = a+g_bar(num_training, l)*z(l);
    end
     f_hat(q) = a/b;
     y(q+1) = 0.3*y(q)+0.6*y(q-1)+g_u(q);
     y_hat(q+1) = 0.3*y(q)+0.6*y(q-1)+f_hat(q);
end

%% Plots and Figures

figure1 = figure('Color', [1 1 1]);
plot(1:701, y, 'b', 1:701, y_hat, 'r:', 'Linewidth', 2); 
legend('output of the plant', 'output of the identification model')
axis([0 701 -5 5]);
grid on

figure2 = figure('Color', [1 1 1]);
xp = -2:0.001:2;
for l = 1:M
    miu_x = exp(-((xp-x_bar(1, l))./(sigma (1,l))).^2);
    plot(xp, miu_x, 'Linewidth', 2);
    hold on
end

xlabel('u');
ylabel('initial MF''s');
axis([-1 1 0 1]);

figure3 = figure('Color', [1 1 1]);
for l = 1:M
    miu_x = exp(-((xp-x_bar(num_training, l))./ (sigma (num_training, l))).^2);
    plot (xp, miu_x, 'Linewidth', 2);
    hold on
end

xlabel('u');
ylabel('final MF''s');
axis ([-1 1 0 1]);
