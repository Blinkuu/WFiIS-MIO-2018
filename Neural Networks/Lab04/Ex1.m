clc; clear;

fileID = fopen('lab_training.txt', 'r');
formatSpec = '%f';
sizeData = [7 Inf];
file = fscanf(fileID, formatSpec, sizeData);

trainData = transpose(file);

fileID = fopen('lab_testing.txt', 'r');
formatSpec = '%f';
sizeData = [7 Inf];
file = fscanf(fileID, formatSpec, sizeData);

testData = transpose(file);

x_train(:,1:4) = trainData(:,1:4);
t_train(:,1:3) = trainData(:,5:7);

x_test(:,1:4) = testData(:,1:4);
t_test(:,1:3) = testData(:,5:7);

x_train = transpose(x_train);
t_train = transpose(t_train);
x_test = transpose(x_test);
t_test = transpose(t_test);

net = feedforwardnet(4);
net = configure(net, x_train, t_train);
net.divideFcn = 'dividetrain';
%net.layers{1}.transferFcn = 'tansig';
net.layers{2}.transferFcn = 'hardlim';
net.trainParam.epochs = 100;
%view(net)
net = train(net, x_train, t_train);
y = net(x_test);

for i = 1:length(t_test)
    val = max(y(:,i));
    for j = 1:3
        if y(j,i) == val
           output(j,i) = 1; 
        else
           output(j,i) = 0;
        end
    end
end

errorsMat = zeros(3,3);
for i = 1:length(t_test)
    if output(1,i) == 1
       column = 1; 
    end

    if output(2,i) == 1
        column = 2; 
    end

    if output(3,i) == 1
       column = 3; 
    end

    %%%%%%%%%%%%%%%%%%%%%

    if t_test(1,i) == 1
       row = 1; 
    end

    if t_test(2,i) == 1
        row = 2; 
    end

    if t_test(3,i) == 1
       row = 3; 
    end

    errorsMat(row,column) = errorsMat(row,column) + 1;
end

errorsMat
perc = (errorsMat(1,1) + errorsMat(2,2) + errorsMat(3,3))/length(t_test)
MSE = perform(net, t_test, output)


surf(x_test, output)
colorbar
