% Time series prediction
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
max_closing = max(closing);
closing = closing/max_closing - 0.5;

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
net = feedforwardnet([3 3 3]);
net = configure(net, x_learn, t_learn);
net.layers{1}.transferFcn = 'tansig';
net.layers{2}.transferFcn = 'tansig';
net.layers{3}.transferFcn = 'tansig';
net.layers{4}.transferFcn = 'tansig';
net.trainFcn = 'trainlm';
net.trainParam.lr = 0.05;
net.trainParam.epochs = 3000;
net.trainParam.goal = 1e-7;
net.divideFcn = 'divideind';
net.divideParam.trainInd = 1:length(x_learn)-365;
net.divideParam.valInd = length(x_learn)-364:length(x_learn);
%view(net)
net = train(net, x_learn, t_learn);

% Neural network working on learning data
%{
y = net(x_learn);

y_temp = (y+0.5) * max_closing;
t_learn_temp = (t_learn+0.5) * max_closing;
closing_temp = (closing+0.5)*max_closing;

MSE = perform(net, t_learn_temp, y_temp)
error_vec = t_learn_temp-y_temp;
max_error = max(abs(error_vec));
relative_error_vec = ((t_learn_temp-y_temp)/max_error)*100;

plot(1:length(y_temp),y_temp);
hold on;
plot(1:length(t_learn_temp), t_learn_temp);
title('Prediction')
xlabel('time')
ylabel('closing value')
legend('Output', 'Target')
hold off;


%plot(1:length(error_vec),error_vec);
%plot(1:length(relative_error_vec),relative_error_vec);

plot(1:length(y_temp), y_temp,'r',1:length(y),t_learn_temp,'b');
hold on;
title('Prediction')
xlabel('time')
ylabel('closing value')
legend('Output', 'Target')
hold off;
%}

% Neural network working on testing data
y = net(x_test);

y_temp = (y+0.5) * max_closing;
t_test_temp = (t_test+0.5) * max_closing;
closing_temp = (closing+0.5)*max_closing;

MSE = perform(net, t_test_temp, y_temp);
error_vec = t_test-y_temp;
max_error = max(abs(error_vec));
relative_error_vec = ((t_test_temp-y_temp)/max_error)*100;

plot(1:length(y_temp),y_temp);
hold on;
plot(1:length(t_test_temp), t_test_temp);
title('Prediction')
xlabel('time')
ylabel('closing value')
legend('Output', 'Target')
hold off;

%plot(1:length(error_vec),error_vec);
%plot(1:length(relative_error_vec),relative_error_vec);

clearvars t_test_temp y_temp closing_temp
