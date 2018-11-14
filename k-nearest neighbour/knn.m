function [error_rate] =  myKNN(training_data, test_data, k)
    
    [m,n] = size(training_data);
    [test_m,test_n] = size(test_data);
    
    train_X = training_data(:,1:n-1);
    train_Y = training_data(:,n);
    
    test_data1 = test_data(:,1:test_n-1);
    
    predicted_class=zeros(test_m,1);
    
    
    for i = 1:test_m
        dist_predicted_class = Inf(k,2);
        for j= 1:m
            dist_ij = norm(test_data1(i,:) - train_X(j,:));
            
            if dist_ij<dist_predicted_class(1,1)
                dist_predicted_class(1,1)=dist_ij;
                dist_predicted_class(1,2)=train_Y(j);
                dist_predicted_class = sortrows(dist_predicted_class,1,'descend');
                pred_array = dist_predicted_class(:,2);
                
            end
            
            predicted_class(i)=mode(pred_array);
            
        end
    end
    
   % Mdl = fitcknn(train_X,train_Y,'NumNeighbors',k);
    %predicted_class = predict(Mdl, test_data1);
    
    error = (predicted_class~=test_data(:,test_n));
    error_rate = sum(error)/test_m;
