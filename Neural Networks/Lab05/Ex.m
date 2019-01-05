clc; clear;

% Import data
filename = 'Dane_lab5.csv';
delimiter = ';';
startRow = 2;
formatSpec = '%s%f%s%f%f%[^\n\r]';
fileID = fopen(filename,'r');
dataArray = textscan(fileID, formatSpec, 'Delimiter', delimiter, 'EmptyValue' ,NaN,'HeaderLines' ,startRow-1, 'ReturnOnError', false);
fclose(fileID);
dataArray([2, 4, 5]) = cellfun(@(x) num2cell(x), dataArray([2, 4, 5]), 'UniformOutput', false);
Danelab5 = [dataArray{1:end-1}];
clearvars filename delimiter startRow formatSpec fileID dataArray ans;

closing = cell2mat(Danelab5(:,2));
for i = 1:3543
   price(i,1) = str2double(Danelab5{i,3});
end

for i = 1:3543
   open(i,1) = str2double(Danelab5{i,4});
end

for i = 1:3543
   daily(i,1) = str2double(Danelab5{i,5});
end

clearvars filename delimiter startRow formatSpec fileID dataArray i ans;

% Preprocessing of data
closing = closing/max(closing) - 0.5;

data_size = length(Danelab5);
t_data = 300;
offset = 5;
for i = 1:offset
    x_learn(i,1:data_size-t_data-offset) = closing(i:data_size-t_data-offset+i-1);
end
t_learn = closing(offset+1:data_size-t_data)';

for i = 1:offset
   x_test(i,1:t_data) = closing(data_size-t_data-offset+i:data_size-offset+i-1);
end
t_test = closing(data_size-t_data+1:data_size)';

clearvars i

% Neural network
net = feedforwardnet([5 3 3]);
net = configure(net, x_learn, t_learn);
net.layers{1}.transferFcn = 'tansig';
net.layers{2}.transferFcn = 'tansig';
net.layers{3}.transferFcn = 'tansig';
net.layers{4}.transferFcn = 'purelin';
net.trainFcn = 'trainlm';
net.trainParam.lr = 0.05;
net.trainParam.epochs = 3000;
net.trainParam.goal = 1e-7;
net.divideFcn = 'divideind';
net.divideParam.trainInd = 1:length(x_learn)-365;
net.divideParam.valInd = length(x_learn)-364:length(x_learn);
net = train(net, x_learn, t_learn);
%{
y = net(x_learn);

for i = 1:length(y)
   error_vec(i) = t_learn(i)-y(i);
end

plot(1:length(error_vec),error_vec)

MSE = perform(net, t_learn, y);

plot(1:length(y), y,'r',1:length(y),closing(1:data_size-t_data-offset),'b');
hold on;
title('Prediction')
xlabel('time')
ylabel('closing value')
legend('Next day stock value', 'Stock value')
hold off;

%y = net(x_learn);
%plot(1:length(y), y, 'r', 1:length(y), closing(1:data_size-t_data-offset))
%}


y = net(x_test);

for i = 1:length(y)
   error_vec(i) = t_test(i)-y(i);
end
plot(1:length(error_vec),error_vec);

max_error = max(abs(error_vec))

for i = 1:length(y)
   relative_error_vec(i) = ((t_test(i)-y(i))/max_error)*100;
end
plot(1:length(error_vec),relative_error_vec);

MSE = perform(net, t_test, y);

plot(1:length(y), y,'r',1:length(y),closing(data_size-t_data+1:data_size),'b');
hold on;
title('Prediction')
xlabel('time')
ylabel('closing value')
legend('Next day stock value', 'Stock value')
hold off;

%y = net(x_learn);
%plot(1:length(y), y, 'r', 1:length(y), closing(1:data_size-t_data-offset))





