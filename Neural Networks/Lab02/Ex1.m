% Neuron
clear; clc;

n = 2; % Dimension
n_learn = 100; % Amount of learning data
n_test = 1000; % Amount of testing data

% Generating Normally Distributed data
K1_learn = [randn(1,n_learn) + 2; randn(1,n_learn)];
K1_test  = [randn(1,n_test) + 2; randn(1,n_test)];

K2_learn = [randn(1,n_learn) - 2; randn(1,n_learn)];
K2_test = [randn(1,n_test) - 2; randn(1,n_test)];

K3_learn = [randn(1,n_learn); randn(1,n_learn) + 2];
K3_test = [randn(1,n_test); randn(1,n_test) + 2];

K4_learn = [randn(1,n_learn); randn(1,n_learn) - 2];
K4_test = [randn(1,n_test); randn(1,n_test) - 2];

% Merging learning and testing data to one vector
x_learn = [K1_learn K2_learn K3_learn K4_learn];
x_test = [K1_test K2_test K3_test K4_test];

k1 = [0 0]';
k2 = [1 1]';
k3 = [0 1]';
k4 = [1 0]';

t_learn = [repmat(k1,1,n_learn) repmat(k2,1,n_learn) repmat(k3,1,n_learn) repmat(k4,1,n_learn)];
t_test = [repmat(k1,1,n_test) repmat(k2,1,n_test) repmat(k3,1,n_test) repmat(k4,1,n_test)];

scatter(K1_learn(1,:), K1_learn(2,:),'.');
hold on;
scatter(K2_learn(1,:), K2_learn(2,:),'.');
scatter(K3_learn(1,:), K3_learn(2,:),'.')
scatter(K4_learn(1,:), K4_learn(2,:),'.');
hold off;
legend('K1', 'K2', 'K3', 'K4');

net = perceptron;
net = configure(net, x_learn, t_learn);
net.trainParam.epochs = 500;
net = train(net, x_learn, t_learn);

y = net(x_test);

error = 0;
for i = 1:4*n_test
   if y(1,i) ~= t_test(1,i) || y(2,i) ~= t_test(2,i)
       error = error + 1;
   end
end

error
perc = 1 - (error/(4*n_test))

scatter(K1_test(1,:), K1_test(2,:),'.');
hold on;
scatter(K2_test(1,:), K2_test(2,:),'.');
scatter(K3_test(1,:), K3_test(2,:),'.')
scatter(K4_test(1,:), K4_test(2,:),'.');
hold off;
legend('K1', 'K2', 'K3', 'K4');
plotpc(net.iw{1, 1},net.b{1});
