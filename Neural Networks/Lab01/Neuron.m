% Neuron
clear; clc;

n = 2; % Dimension
n_K1_learn = 100; % Amount of learning data
n_K1_test = 1000; % Amount of testing data

n_K2_learn = 100; % Amount of learning data
n_K2_test = 1000; % Amount of testing data

% Generating Normally Distributed data
K1_learn = randn(n_K1_learn, n);
K1_test = randn(n_K1_test, n);

K2_learn = randn(n_K2_learn, n);
K2_test = randn(n_K2_test, n);

for i = 1:n_K2_learn
    K2_learn(i,1) = K2_learn(i,1) + 2.0;
end

for i = 1:n_K2_test
    K2_test(i,1) = K2_test(i,1) + 2.0;
end

% Merging learning data to one vector
x = [K1_learn', K2_learn']; 

% Output I want
for i = 1:(n_K1_learn + n_K2_learn)
    if i <= n_K1_learn
        t(i) = 1;
    else
        t(i) = 0;
    end
end

net = perceptron;
%view(net);

% Merging test data to one vector
test_x = [K1_test', K2_test'];

% Calculating mean percentage
perc = 0;
for k = 1:10
    net = train(net, x, t);
    y = net(test_x);

    for i = 1:(n_K1_test + n_K2_test)
        if i <= n_K1_test
            t_test(i) = 1;
        else
            t_test(i) = 0;
        end
    end

    sum = 0;

    for i = 1:(n_K1_test + n_K2_test)
        if y(i) == t_test(i)
            sum = sum + 1;
        end
    end

    percentage = sum/(n_K1_test + n_K2_test)
    perc = perc + percentage;
end

perc = perc/10


plot(K2_test, K1_test, '.')

weights = net.iw{1,1};
bias = net.b{1};

plotpc(weights, bias)

% Failure matrix
failureMatrix = [0 0; 0 0];

for i = 1:(n_K1_test + n_K2_test)
   if i <= n_K1_test
      if  y(i) == t_test(i)
          failureMatrix(1,1) = failureMatrix(1,1) + 1;
      else 
          failureMatrix(2,1) = failureMatrix(2,1) + 1;
      end
   else    
      if  y(i) == t_test(i)
          failureMatrix(2,2) = failureMatrix(2,2) + 1;
      else 
          failureMatrix(1,2) = failureMatrix(1,2) + 1;
      end
   end
end

failureMatrix
