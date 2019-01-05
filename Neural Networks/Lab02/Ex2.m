% Neuron
clear; clc;

load iris_dataset

x = irisInputs;
t = irisTargets;

%Preparing matrix for K-Fold Cross Validation
for i = 1:150
    if i <= 50
       t_string(i) = "virginica";
       t_2(1,i) = 1;
       t_2(2,i) = 1;
    end

    if i > 50  && i <= 100
       t_string(i) = "setosa"; 
       t_2(1,i) = 0;
       t_2(2,i) = 1;
    end
    
    if i > 100
       t_string(i) = "versicolor"; 
       t_2(1,i) = 1;
       t_2(2,i) = 0;
    end
end

k = 6;
indices = crossvalind('Kfold',t_string,k);

meanPerc = 0;
for i = 1:k
    learn_c = 1;
    test_c = 1;
    for j = 1:150
        if indices(j) == i
            x_learn(:,learn_c) = x(:,j);
            t_learn(:,learn_c) = t(:,j);
            t_learn2(:,learn_c) = t_2(:,j);
            learn_c = learn_c + 1;
        else
            x_test(:,test_c) = x(:,j);
            t_test(:,test_c) = t(:,j); 
            t_test2(:,test_c) = t_2(:,j);
            test_c = test_c + 1;
        end
    end
    
    %net = perceptron;
    %net = configure(net, x_learn, t_learn);
    %view(net)
    %net.trainParam.epochs = 500;
    %net = train(net, x_learn, t_learn);
    %y = net(x_test);
    
    net = perceptron;
    net = configure(net, x_learn, t_learn2);
    %view(net)
    net.trainParam.epochs = 500;
    net = train(net, x_learn, t_learn2);
    y = net(x_test);
      
    error = 0;
    %for i = 1:length(y)
    %   if ~isequal(y(1,i), t_test(1,i)) || ~isequal(y(2,i), t_test(2,i)) || ~isequal(y(3,i), t_test(3,i))
    %       error = error + 1;
    %   end
    %end
    
    for i = 1:length(y)
       if (y(1,i) ~= t_test2(1,i)) || (y(2,i) ~= t_test2(2,i))
           error = error + 1;
       end
    end

    error
    perc = 1 - error/length(y)
    meanPerc = meanPerc + perc;
end

meanPerc = meanPerc/k
