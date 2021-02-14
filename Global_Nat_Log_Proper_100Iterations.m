%%% STOCHASTIC GRADIENT DESCENT %%%%%%%%
%% Initializing variables . . .
alpha = 0.001;
iterations = 100;
theta0_vals = zeros (1, iterations);
theta1_vals = zeros (1, iterations);
theta = [0 ; 0];
J_vals = zeros (iterations);
J_history = zeros (iterations, 1);
theta_store = zeros (2, iterations);

%% Running Iterations . . . 
for i = 1:iterations
h = X_train * theta;
errors = h - GDP_train;
theta_change = (alpha * (X_train' * errors)) / m;
theta = theta - theta_change;
theta0_vals (i) = theta (1, :);
theta1_vals (i) = theta (2, :);
theta_store (:, i) = theta;
J_history(i) = (sum ((h - GDP_train).^2))/ (2 * m);
J_vals (i, i) = (sum ((h - GDP_train).^2))/ (2 * m);
end
%% End of Iterations.

SGD_Loss = J_history;
SGD_theta0 = theta0_vals;
SGD_theta1 = theta1_vals;
SGD_thetaStore = theta_store;
SGD_theta = theta;

min (SGD_Loss)
max (SGD_Loss)
SGD_theta
%%% END OF STOCHASTIC GRADIENT DESCENT %%%%

%%% STOCHASTIC GRADIENT DESCENT WITH MOMENTUM %%%%%%%%
%% Initializing variables . . .
alpha = 0.001;
iterations = 100;
theta0_vals = zeros (1, iterations);
theta1_vals = zeros (1, iterations);
theta = [0 ; 0];
J_vals = zeros (iterations);
J_history = zeros (iterations, 1);
theta_store = zeros (2, iterations);

%% Initializing moments . . . .
momentum = [0; 0]; % Momentum
momentum_store = zeros (2, iterations);

%% Initializing decays . . . .
beta = 0.9;

%% Running Iterations . . . 
for i = 1:iterations
h = X_train * theta;
errors = h - GDP_train;
momentum = (beta * momentum) + ((X_train' * errors) / m);
theta_change = (alpha * momentum);
theta = theta - theta_change;
momentum_store (:, i) = momentum;
theta0_vals (i) = theta (1, :);
theta1_vals (i) = theta (2, :);
theta_store (:, i) = theta;
J_history(i) = (sum ((h - GDP_train).^2))/ (2 * m);
J_vals (i, i) = (sum ((h - GDP_train).^2))/ (2 * m);
end
%% End of Iterations.

SGDwM_Loss = J_history;
SGDwwM_theta0 = theta0_vals;
SGDwM_theta1 = theta1_vals;
SGDwM_thetaStore = theta_store;
SGDwM_theta = theta;

min (SGDwM_Loss)
max (SGDwM_Loss)
SGDwM_theta
%%% END OF STOCHASTIC GRADIENT DESCENT WITH MOMENTUM %%%

%%% STOCHASTIC GRADIENT DESCENT NESTEROV %%%%%%%%
%% Initializing variables . . .
alpha = 0.001;
iterations = 100;
theta0_vals = zeros (1, iterations);
theta1_vals = zeros (1, iterations);
theta = [0 ; 0];
J_vals = zeros (iterations);
J_history = zeros (iterations, 1);
theta_store = zeros (2, iterations);

%% Initializing moments . . . .
momentum = [0; 0]; % Momentum
momentum_store = zeros (2, iterations);

%% Initializing decays . . . .
beta = 0.9;

%% Running Iterations . . . 
for i = 1:iterations
lambda = theta - (beta * momentum);  
h = X_train * lambda;
errors = h - GDP_train;
momentum = (beta * momentum) + ((X_train' * errors) / m);
theta_change = (alpha * momentum);
theta = theta - theta_change;
momentum_store (:, i) = momentum;
theta0_vals (i) = theta (1, :);
theta1_vals (i) = theta (2, :);
theta_store (:, i) = theta;
J_history(i) = (sum ((h - GDP_train).^2))/ (2 * m);
J_vals (i, i) = (sum ((h - GDP_train).^2))/ (2 * m);
end
%% End of Iterations.

SGDNest_Loss = J_history;
SGDNest_theta0 = theta0_vals;
SGDNest_theta1 = theta1_vals;
SGDNest_thetaStore = theta_store;
SGDNest_theta = theta;

min (SGDNest_Loss)
max (SGDNest_Loss)
SGDNest_theta
%%% END OF STOCHASTIC GRADIENT DESCENT NESTEROV %%%%%%

%%%%%% STOCHASTIC GRADIENT DESCENT RMS PROPAGATION %%%%%%%
%% Initializing variables . . .
alpha = 0.001;
iterations = 100;
theta0_vals = zeros (1, iterations);
theta1_vals = zeros (1, iterations);
theta = [0 ; 0];
J_vals = zeros (iterations);
J_history = zeros (iterations, 1);
theta_store = zeros (2, iterations);

%% Initializing moments . . . .
momentum = [0; 0]; % Momentum
momentum_store = zeros (2, iterations);

%% Initializing decays . . . .
beta = 0.9;

%% Running Iterations . . . 
for i = 1:iterations
h = X_train * theta;
errors = h - GDP_train;
momentum = (beta * momentum) + (1 - beta) * (((X_train' * errors) / m).^2);
theta_change = pinv (sqrt (abs (momentum)) + 10e-8) * (alpha * ((X_train' * errors) / m));
theta = theta - theta_change;
momentum_store (:, i) = momentum;
theta0_vals (i) = theta (1, :);
theta1_vals (i) = theta (2, :);
theta_store (:, i) = theta;
J_history(i) = (sum ((h - GDP_train).^2))/ (2 * m);
J_vals (i, i) = (sum ((h - GDP_train).^2))/ (2 * m);
end
%% End of Iterations.

RMSProp_Loss = J_history;
RMSProp_theta0 = theta0_vals;
RMSProp_theta1 = theta1_vals;
RMSProp_thetaStore = theta_store;
RMSProp_theta = theta;

min (RMSProp_Loss)
max (RMSProp_Loss)
RMSProp_theta
%%% END OF STOCHASTIC GRADIENT DESCENT RMS PROPAGATION %%%%%%

%%%%%% STOCHASTIC GRADIENT DESCENT ADAM %%%%%%%
%% Initializing variables . . .
alpha = 0.001;
iterations = 100;
theta0_vals = zeros (1, iterations);
theta1_vals = zeros (1, iterations);
theta = [0 ; 0];
J_vals = zeros (iterations);
J_history = zeros (iterations, 1);
theta_store = zeros (2, iterations);

%% Initializing moments . . . .
moment_1 = [0; 0]; % First moment
moment_2 = [0; 0]; % Second moment
m1_store = zeros (2, iterations);
m2_store = zeros (2, iterations);

%% Initializing decays . . . .
beta_1 = 0.9;
beta_2 = 0.999;

%% Running Iterations . . . 
for i = 1:iterations
h = X_train * theta;
errors = h - GDP_train;
moment_1 = (beta_1 * moment_1) + (1 - beta_1) * ((X_train' * errors) / m);
moment_2 = (beta_2 * moment_2) + (1 - beta_2) * ((X_train' * errors) / m);
theta_change = pinv (sqrt (abs (moment_2)) + 10e-8) * (alpha * moment_1);
theta = theta - theta_change;
m1_store (:, i) = moment_1;
m2_store (:, i) = moment_2;
theta0_vals (i) = theta (1, :);
theta1_vals (i) = theta (2, :);
theta_store (:, i) = theta;
J_history(i) = (sum ((h - GDP_train).^2))/ (2 * m);
J_vals (i, i) = (sum ((h - GDP_train).^2))/ (2 * m);
end
%% End of Iterations.

Adam_Loss = J_history;
Adam_theta0 = theta0_vals;
Adam_theta1 = theta1_vals;
Adam_thetaStore = theta_store;
Adam_theta = theta;

min (Adam_Loss)
max (Adam_Loss)
Adam_theta
%%% END OF STOCHASTIC GRADIENT DESCENT ADAM %%%%%%


GDP_SGD_cap = X_test * SGD_theta;
GDP_SGDwM_cap = X_test * SGDwM_theta;
GDP_SGDNest_cap = X_test * SGDNest_theta;
GDP_RMSProp_cap = X_test * RMSProp_theta;
GDP_Adam_cap = X_test * Adam_theta;

%%% Displaying the Data
GDP_SGD_cap
GDP_SGDwM_cap
GDP_SGDNest_cap
GDP_RMSProp_cap
GDP_Adam_cap

%%% RUN STUDENT'S T-TEST N DATA %%%%%
pkg load statistics
[h, pval, ci, stats] = ttest2 (GDP_test', GDP_SGD_cap')
[h, pval, ci, stats] = ttest2 (GDP_test', GDP_SGDwM_cap')
[h, pval, ci, stats] = ttest2 (GDP_test', GDP_SGDNest_cap')
[h, pval, ci, stats] = ttest2 (GDP_test', GDP_RMSProp_cap')
[h, pval, ci, stats] = ttest2 (GDP_test', GDP_Adam_cap')

theta_norm_eqn = pinv (X_train' * X_train) * X_train' * GDP_train
Normal_Eqn_cap = X_test * theta_norm_eqn;
[h, pval, ci, stats] = ttest2 (GDP_test', Normal_Eqn_cap')

GDP_test
GDP_train

Pop_test
Pop_train

m_test
%%%% Root Mean Square Error Test
RMSE_SGD = sqrt (sum((GDP_SGD_cap - GDP_test).^2)/m_test)
RMSE_SGDwM = sqrt (sum((GDP_SGDwM_cap - GDP_test).^2)/m_test)
RMSE_SGDNest = sqrt (sum((GDP_SGDNest_cap - GDP_test).^2)/m_test)
RMSE_RMSProp = sqrt (sum((GDP_RMSProp_cap - GDP_test).^2)/m_test)
RMSE_Adam = sqrt (sum((GDP_Adam_cap - GDP_test).^2)/m_test)
RMSE_Normal_Eqn = sqrt (sum((Normal_Eqn_cap - GDP_test).^2)/m_test)

%%% Loss Values
epoch = [1:iterations];
Loss_Data = [SGD_Loss, SGDwM_Loss, SGDNest_Loss, RMSProp_Loss, Adam_Loss];
save 100Loss_Data.txt Loss_Data

%%%%% Save Loss Values
save 100SGD_Loss.txt SGD_Loss
save 100SGDwM_Loss.txt SGDwM_Loss
save 100SGDNest_Loss.txt SGDNest_Loss
save 100RMSProp_Loss.txt RMSProp_Loss
save 100Adam_Loss.txt Adam_Loss

%%%% Save Estimated Values
Estimated_Data100 = [GDP_SGD_cap, GDP_SGDwM_cap, GDP_SGDNest_cap, GDP_RMSProp_cap, GDP_Adam_cap, Normal_Eqn_cap]
save Estimated_Data100.txt Estimated_Data100

%%%%% Plotting Loss
plot (epoch, SGD_Loss, '-', 'Color', 'black')
hold on;
plot (epoch, SGDwM_Loss, '-', 'Color', 'black')
plot (epoch, RMSProp_Loss, '-', 'Color', 'black')
plot (epoch, Adam_Loss, '-', 'Color', 'black')
xlabel('Iteration'); ylabel('Loss');

plot (epoch, SGD_Loss, '-', 'Color', 'blue')
hold on;
plot (epoch, SGDwM_Loss, '-', 'Color', 'green')
plot (epoch, RMSProp_Loss, '-', 'Color', 'red')
plot (epoch, Adam_Loss, '-', 'Color', 'black')
xlabel('Iteration'); ylabel('Loss');

plot (epoch, SGD_Loss, '-o', 'Color', 'blue')
hold on;
plot (epoch, SGDwM_Loss, '-o', 'Color', 'green')
plot (epoch, RMSProp_Loss, '-o', 'Color', 'red')
plot (epoch, Adam_Loss, '-o', 'Color', 'black')
xlabel('Iteration'); ylabel('Loss');

plot (epoch, SGD_Loss, '->', 'Color', 'black')
hold on;
plot (epoch, SGDwM_Loss, '-+', 'Color', 'black')
plot (epoch, RMSProp_Loss, '-*', 'Color', 'black')
plot (epoch, Adam_Loss, '-o', 'Color', 'black')
xlabel('Iteration'); ylabel('Loss');

theta_r = [11.96333 ; 0.79036];
GDP_reg_cap = X_test * theta_r;
[h, pval, ci, stats] = ttest2 (GDP_test', GDP_reg_cap')
RMSE_GDP_reg = sqrt (sum((GDP_reg_cap - GDP_test).^2)/m_test)