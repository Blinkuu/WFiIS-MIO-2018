% Neuron
clear; clc;

n = 2; % Dimension
n_learn = 50; % Amount of learning data
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

k1 = [-1 -1]';
k2 = [1 1]';
k3 = [-1 1]';
k4 = [1 -1]';

t_learn = [repmat(k1,1,n_learn) repmat(k2,1,n_learn) repmat(k3,1,n_learn) repmat(k4,1,n_learn)];
t_test = [repmat(k1,1,n_test) repmat(k2,1,n_test) repmat(k3,1,n_test) repmat(k4,1,n_test)];

scatter(K1_learn(1,:), K1_learn(2,:),'.');
hold on;
scatter(K2_learn(1,:), K2_learn(2,:),'.');
scatter(K3_learn(1,:), K3_learn(2,:),'.')
scatter(K4_learn(1,:), K4_learn(2,:),'.');
hold off;
legend('K1', 'K2', 'K3', 'K4');

net = linearlayer(0,0.04);
net.trainFcn = 'traingd';
net = train(net, x_learn, t_learn);
y = net(x_test);

errorMat = zeros(4,4);

for i = 1:4*n_test
   if y(1,i) < 0 && y(2,i) < 0
       column = 1;
   end
   
   if y(1,i) > 0 && y(2,i) > 0
       column = 2;
   end
   
   if y(1,i) < 0 && y(2,i) > 0 
       column = 3;
   end
   
   if y(1,i) > 0 && y(2,i) < 0
       column = 4;
   end
   
   %%%%%%%%%%%%%%%%%%%%%%%%
   
   if t_test(1,i) < 0 && t_test(2,i) < 0
       row = 1;
   end
   
   if t_test(1,i) > 0 && t_test(2,i) > 0
       row = 2;
   end
   
   if t_test(1,i) < 0 && t_test(2,i) > 0 
       row = 3;
   end
   
   if t_test(1,i) > 0 && t_test(2,i) < 0
       row = 4;
   end
   
   errorMat(row,column) = errorMat(row,column) + 1;
   
end

net.IW{1}

errorMat
perc = (errorMat(1,1) + errorMat(2,2) + errorMat(3,3) + errorMat(4,4))/(4*n_test)

%scatter(K1_learn(1,:), K1_learn(2,:),'.');
%hold on;
%scatter(K2_learn(1,:), K2_learn(2,:),'.');
%scatter(K3_learn(1,:), K3_learn(2,:),'.')
%scatter(K4_learn(1,:), K4_learn(2,:),'.');
%hold off;
%legend('K1', 'K2', 'K3', 'K4');

scatter(K1_test(1,:), K1_test(2,:),'.');
hold on;
scatter(K2_test(1,:), K2_test(2,:),'.');
scatter(K3_test(1,:), K3_test(2,:),'.')
scatter(K4_test(1,:), K4_test(2,:),'.');
hold off;
legend('K1', 'K2', 'K3', 'K4');
plotpc(net.iw{1, 1},net.b{1});
