clc; clear;

A = 6; %4
B = 3; %3
C = 10; %1

D = 8; %3
E = 2; %2
F = 56; %1

X1 = 0;
X2 = 10;
N = 1000;

X = linspace(X1, X2, N);

for i = 1:N
   Y(i) = A*sin(B*X(i) + C) + D*cos(E*X(i) + F);
end

n_learn = 100;
n_test = N;

for i = 1:n_learn
   rnd = randi([1 N],1,1);
   x_learn(i) = X(rnd);
   y_learn(i) = Y(rnd);
end

x_test = X;
y_test = Y;

plot(X,Y,'r')
hold on;
title('Target function')
xlabel('X')
ylabel('Y')
%legend('y = 6sin(3x + 10) + 8cos(2x + 56)')
scatter(x_learn(1,:), y_learn(1,:),'*');
legend('y = 6sin(3x + 10) + 8cos(2x + 56)', 'Random learning data')
hold off;

net = feedforwardnet(10);
%net = configure(net, x_learn, y_learn);

net.trainParam.epochs = 1000;
net.divideParam.trainRatio = 1.0;
net.divideParam.valRatio = 0.0;
net.divideParam.testRatio = 0.0;
net.divideParam

net.layers{1}.transferFcn = 'tansig';
%net.layers{1}.transferFcn = 'logsig';

net = train(net, x_learn, y_learn);

output = net(x_test);
MSE = perform(net, Y, output)
plot(X,Y,'r')
hold on;
plot(x_test, output, 'b')
title('Function Approximation')
xlabel('X')
ylabel('Y')
legend('Target function', 'Output')
hold off;