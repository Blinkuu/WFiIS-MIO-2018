% Neuron
clear; clc;

fileID = fopen('Lab1_training.txt', 'r');
formatSpec = '%f';
sizeData = [4 Inf];
file = fscanf(fileID, formatSpec, sizeData);

Data = transpose(file);

k = 3;

dataToCross = Data(1:30,4);
indices = crossvalind('KFold', dataToCross, k);

percentageArr = [0];
for cur_learning = 1:k
    x_learn = [0];
    t_learn = [0];
    x = [0];
    t = [0];
    learn_iter = 1;
    test_iter = 1;
    for i = 1:30
        if cur_learning == indices(i)
            for j = 1:3
                x_learn(learn_iter,j) = Data(i,j);
            end
            t_learn(learn_iter) = Data(i,4);
            learn_iter = learn_iter + 1;
        else
            for j = 1:3
                x(test_iter,j) = Data(i,j);
            end
            t(test_iter) = Data(i,4);
            test_iter = test_iter + 1;
        end
    end

    x_learn = transpose(x_learn);
    x = transpose(x);

    net = perceptron;
    net = train(net, x_learn, t_learn);

    %view(net);

    y = net(x);

    sum = 0;
    for i = 1:30 - (30/k) - 1
       if  y(i) == t(i)
          sum = sum + 1; 
       end
    end
    
    percentage = sum/(30 - (30/k) - 1);
    percentageArr(cur_learning) = percentage;
end

percentageArr
meanPercentage = mean(percentageArr)
stdPercentage = std(percentageArr)

%plot(Data, '.');
%weights = net.iw{1,1,1};
%bias = net.b{1};
%plotpc(weights, bias);
