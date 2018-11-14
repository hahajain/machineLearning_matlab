function[] = MultiGaussian(training_data, test_data, model)
    
    [m,n] = size(training_data);
    [test_m,test_n] = size(test_data);
    
    test_data1 = test_data(test_data(:,test_n)==1,1:test_n-1);
    train_class1 = training_data(training_data(:,n)==1,1:n-1);
    train_class2 = training_data(training_data(:,n)==2,1:n-1);
    
    [m1,n1] =size(train_class1); %size of class 1
    [m2,n2] =size(train_class2); %size of class 2
    
    pc1 = m1/(m1+m2)
    pc2 = 1-pc1
    
    u1 = mean(train_class1,1)
    u2 = mean(train_class2,1)
    
    C1 = cov(train_class1);
    C2 = cov(train_class2);
    
    predicted_class = zeros(test_m,1);
    %Case1 Independent variances
   
    switch model
       case 1
            for i = 1:test_m
        
            
                pcx1 =  - 0.5 * log(det(C1)) - 0.5 * (test_data(i,1:test_n-1)-u1) * inv(C1) * (test_data(i,1:test_n-1)-u1)' + log(pc1);
                pcx2 =  - 0.5 * log(det(C2)) - 0.5 * (test_data(i,1:test_n-1)-u2) * inv(C2) * (test_data(i,1:test_n-1)-u2)' + log(pc2);
        
        
                if pcx1 - pcx2 >=0
                    predicted_class(i) = 1;
                else
                    predicted_class(i) = 2;
                end
        
            end
    
            error = (predicted_class~=test_data(:,test_n));
            error_rate = sum(error)/test_m;
            
            disp("Part A result:");
            disp("S1:");
            disp(C1);
            disp("S2:");
            disp(C2);
            disp(sprintf("Part A Error Rate: %f", error_rate));
            fprintf("\n");
        
    %Case2 same variances
       case 2
    
            s = pc1*C1+pc2*C2;
    
            for i = 1:test_m
            
                pcx1 = - 0.5 * log(det(s)) - 0.5 * (test_data(i,1:test_n-1)-u1) * inv(s) * (test_data(i,1:test_n-1)-u1)' + log(pc1);
                pcx2 = - 0.5 * log(det(s)) - 0.5 * (test_data(i,1:test_n-1)-u2) * inv(s) * (test_data(i,1:test_n-1)-u2)' + log(pc2);
        
        
                if pcx1 - pcx2 >= 0
                    predicted_class(i) = 1;
                else
                    predicted_class(i) = 2;
                end
        
            end
    
            error = (predicted_class~=test_data(:,test_n));
            error_rate = sum(error)/test_m;
            
            disp("Part B result:");
            disp("S1:");
            disp(s);
            disp("S2:");
            disp(s);
            disp(sprintf("Part B Error Rate: %f", error_rate));
            fprintf("\n");
    
    %Case 3
       case 3
    
            C1_diag=sum(diag(C1))/size(C1,1);
            C2_diag=sum(diag(C2))/size(C2,1);
    
            alpha1=diag(ones(8,1) * C1_diag);
            alpha2=diag(ones(8,1) * C2_diag);
    
            for i = 1:test_m
            
                pcx1 = - 0.5 * log(det(alpha1)) - 0.5 * (test_data(i,1:test_n-1)-u1) * inv(alpha1) * (test_data(i,1:test_n-1)-u1)' + log(pc1);
                pcx2 = - 0.5 * log(det(alpha2)) - 0.5 * (test_data(i,1:test_n-1)-u2) * inv(alpha2) * (test_data(i,1:test_n-1)-u2)' + log(pc2);
        
                if pcx1 - pcx2 >= 0
                    predicted_class(i) = 1;
                else
                    predicted_class(i) = 2;
                end
        
            end
    
            error = (predicted_class~=test_data(:,test_n));
            error_rate = sum(error)/test_m;
            
            disp("Part C result:");
            disp("Alpha1:");
            disp(alpha1);
            disp("Alpha2:");
            disp(alpha2);
            disp(sprintf("Part C Error Rate: %f", error_rate));
            fprintf("\n");
   end
