%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% reading, pre-processing, and division of data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% setting seed for repeatability of results
rng(3);

disp("Loading data from XLSX file");
data = readtable("diagnosis.xlsx");
data = table2array(data);

% regularization of temperature data
data(:, 1) = (data(:, 1) - mean(data(:, 1)))/std(data(:, 1));

% determining number of records and features
num_records = size(data, 1);
num_features = size(data, 2) - 1;

% fixing the labels to be 1 (TRUE) or -1 (FALSE)
labels = data(:, end);
labels(labels == 0) = -1;
data(:, end) = labels;

% modifying binary variables from [0, 1] to [-1, 1]
data(data == 0) = -1;

% separtating the binary classes
positive_indices = ismember(data(:, end), ones(num_records, 1));
negative_indices = ismember(data(:, end), -1 * ones(num_records, 1));
true_data = data(positive_indices, :);
false_data = data(negative_indices, :);

disp(" ");
disp("Data pre-processed. Example row of data: ");
disp("Features: ");
disp(data(10, 1: end - 1));
disp("Label: " + num2str(data(10, end)));

disp(" ");
disp("Creating train and test set");

% choosing number of data records from each binary class to create the train set
num_choose = 26;
train_set = [true_data(1: num_choose, :);
             false_data(1: num_choose, :)];
train_set = train_set(randperm(size(train_set, 1)), :);

% create the test set similar to the train set
test_set = [true_data(num_choose + 1: num_choose * 2, :);
             false_data(num_choose + 1: num_choose * 2, :)];
test_set = test_set(randperm(size(test_set, 1)), :);

% randomizing the train set before dividing for linear and non-linear classifier
train_set = train_set(randperm(size(train_set, 1)), :);

% dividing into two sets
train_set_a = train_set(1: num_choose, :);
train_set_b = train_set(num_choose + 1: num_choose * 2, :);

disp("Divided training set into two parts: " + num2str(num_choose) + ...
     " records for linear SVM and RBF kernel SVM each");

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% using the primal method to find the best separating hyperplane
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp(" ");
disp("Solving primal optimization problem for the linear SVM...");

% using quadratic programming to solve the soft-margin classifier based on hinge loss
bias_term = -10;
bias = bias_term * ones(size(train_set_a, 1), 1);
features = [train_set_a(:, 1: end - 1) bias];
labels = train_set_a(:, end);

% quadratic part
lambda = 1;
num_weights = num_features + 1;
num_zeta = size(train_set_a, 1);
num_dec_vars = num_weights + num_zeta;
H = zeros(num_dec_vars);
H(1: num_weights, 1: num_weights) = lambda/2 * eye(num_weights);

% linear part
f = 1/num_zeta * [zeros(num_weights, 1); ones(num_zeta, 1)];

% constraints
A = -[[labels .* features eye(num_zeta)]; [zeros(num_zeta, num_weights) eye(num_zeta)]];
b = -[ones(num_zeta, 1) zeros(num_zeta, 1)];

% solving the optimization problem
x = quadprog(H, f, A, b);
w = x(1: num_weights);

disp(" ");
disp("Optimization problem solved with hyperplane weights: ");
disp(w);

% inference on testing set
test_features = [test_set(:, 1: end - 1) bias_term * ones(size(test_set, 1), 1);];
test_labels = test_set(:, end);
results = test_features * w;

% evaluating accuracy
results(results >= 0) = 1;
results(results < 0) = -1;
results_a = results == test_labels;
test_accuracy_a = sum(results_a, 'all')/size(test_labels, 1) * 100;

disp(" ");
disp("[Linear SVM] Accuracy on testing set: " + num2str(test_accuracy_a) + "%");

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% using the kernel trick to find the best separating hypersurface
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% parameters: lambda for soft margins, and sigma for RBF kernel
lambda = 0.005;
sigma = 0.25;

disp(" ");
disp("Solving using the kernel method with lambda = " + num2str(lambda) + ...
     " and sigma =  " + num2str(sigma));

% features, labels, and number of records
num_weights = num_features + 1;
num_zeta = size(train_set_b, 1);
features = train_set_b(:, 1: end - 1);
labels = train_set_b(:, end);

% using RBF kernel
sqr = sum(features' .^ 2);
K = exp((2 * (features * features') - sqr' * ones(1, num_zeta)...
        - ones(num_zeta, 1) * sqr)/(2 * sigma ^ 2));

% label vector included 
% resulting in the quadratic part of the cost function
H = (labels * labels') .* K;

% linear part
f = -ones(num_zeta, 1);

% equality constraint
Aeq = labels';
beq = zeros(1, 1);

% bounds
lb = zeros(num_zeta, 1);
ub = 1/(2 * num_zeta * lambda) * ones(num_zeta, 1);

% solving the quadratic problem
c = quadprog(H, f, [], [], Aeq, beq, lb, ub);

disp(" ");
disp("Optimization problem solved with resulting decision variables values: ");
disp(c);

% for bias calculation
[~, ind] = max(c);
xj = features(ind, :);
b = exp((2 * features * xj' - sum(xj .^ 2) - sqr')/(2 * sigma ^ 2)) .* labels .* c;
b = sum(b) - labels(ind);

% inference on testing set
test_features = test_set(:, 1: end - 1);
test_labels = test_set(:, end);
num_test = size(test_set, 1);
sqr_t = sum(test_features' .^ 2);
results = exp((2 * test_features * features' - sqr_t' * ...
        ones(1, num_zeta) - ones(num_test, 1) * sqr)/(2 * sigma ^ 2));
results = results .* labels' .* c';
results = sum(results, 2) - b;

% evaluating accuracy
results(results >= 0) = 1;
results(results < 0) = -1;
results_b = results == test_labels;
test_accuracy_b = sum(results_b, 'all')/size(test_labels, 1) * 100;

disp(" ");
disp("[Kernel Method] Accuracy on testing set: " + num2str(test_accuracy_b) + "%");