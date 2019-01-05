% Neuron
clear; clc;

load iris_dataset

x = irisInputs;
t = irisTargets;

max = max(x,[],'all');
min = min(x,[],'all');

%Preparing matrix for K-Fold Cross Validation
for i = 1:150
    if i <= 50
       t_string(i) = "virginica";
       t_2(1,i) = -1;
       t_2(2,i) = -1;
    end

    if i > 50  && i <= 100
       t_string(i) = "setosa"; 
       t_2(1,i) = -1;
       t_2(2,i) = 1;
    end
    
    if i > 100
       t_string(i) = "versicolor"; 
       t_2(1,i) = 1;
       t_2(2,i) = -1;
    end
end

k = 5;
indices = crossvalind('Kfold',t_string,k);

meanPerc = 0;
for i = 1:k
    learn_c = 1;
    test_c = 1;
    for j = 1:150
        if indices(j) == i
            x_learn(:,learn_c) = x(:,j);
            t_learn(:,learn_c) = t(:,j);
            t_2_learn(:,learn_c) = t_2(:,j);
            learn_c = learn_c + 1;
        else
            x_test(:,test_c) = x(:,j);
            t_test(:,test_c) = t(:,j);
            t_2_test(:,test_c) = t_2(:,j);
            test_c = test_c + 1;
        end
    end
    
    net = linearlayer(0,0.04);
    net.trainFcn = 'traingd';
 
    % For 2D output
    %net = train(net, x_learn, t_2_learn);
 
    net = train(net, x_learn, t_learn);
    y = net(x_test);
    
    % For 2D output, doesn't really change anything
    %{
    for i = 1:length(t_2_test)
       for j = 1:2
          if y(j,i) >= 0
              output(j,i) = 1;
          else
              output(j,i) = -1;
          end
       end
    end
    
    error = 0;
    for i = 1:length(y)
       if (output(1,i) ~= t_2_test(1,i)) || (output(2,i) ~= t_2_test(2,i))
           error = error + 1;
       end
    end
    
    perc = 1 - error/length(output)
    %}
    
    for i = 1:length(t_test)
        val = y(1,i);
        for p = 1:3
           if y(p,i) > val
              val = y(p,i); 
           end
        end
               
        for j = 1:3
            if y(j,i) == val
               output(j,i) = 1; 
            else
               output(j,i) = 0;
            end
        end
    end

    errorMat = zeros(3,3);
    for i = 1:length(y)
       if output(1,i) == 1 && output(2,i) == 0 && output(3,i) == 0
           column = 1;
       end

       if output(2,i) == 1 && output(1,i) == 0 && output(3,i) == 0
           column = 2;
       end

       if output(3,i) == 1 && output(1,i) == 0 && output(2,i) == 0
           column = 3;
       end

       %%%%%%%%%%%%%%%%%%%%%%%%

       if t_test(1,i) == 1 && t_test(2,i) == 0 && t_test(3,i) == 0
           row = 1;
       end

       if t_test(2,i) == 1 && t_test(1,i) == 0 && t_test(3,i) == 0
           row = 2;
       end

       if t_test(3,i) == 1 && t_test(1,i) == 0 && t_test(2,i) == 0
           row = 3;
       end
       
       errorMat(row,column) = errorMat(row,column) + 1;
    end
    
    errorMat
    perc = (errorMat(1,1) + errorMat(2,2) + errorMat(3,3))/length(output)
    meanPerc = meanPerc + perc;
    
end

meanPerc = meanPerc / k




 