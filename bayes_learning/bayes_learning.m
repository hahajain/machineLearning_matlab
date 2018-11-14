function[p1, p2, pc1, pc2] = Bayes_Learning(training_data , validation_data)
    
    s = "";
    fprintf(" P1            P2          Error rate\n");
    
    [m,n] = size(training_data);

    train_class1 = training_data(training_data(:,n)==1,1:n-1);
    train_class2 = training_data(training_data(:,n)==2,1:n-1);

    [m1,n1] =size(train_class1); %size of class 1
    [m2,n2] =size(train_class2); %size of class 2

    p1= sum(1.-train_class1)/m1; %p1 vector
    p2= sum(1.-train_class2)/m2; %p2 vector
    
    %priors
    

    [val_m, val_n] = size(validation_data);

    predicted_class = zeros(val_m,1);
    
    best_pc1 = 0;
    min_error = 1;
    
    for sig = -5:5
        
        prior = 1/(1+exp(-sig));

        for i = 1:val_m
            pxc1 = 1;
            pxc2 = 1;
            for j = 1:val_n - 1
                pxc1 = pxc1*power(p1(j),1-validation_data(i,j))*power(1-p1(j),validation_data(i,j));
                pxc2 = pxc2*power(p2(j),1-validation_data(i,j))*power(1-p2(j),validation_data(i,j));
            end
            
            if prior*pxc1 > (1-prior)*pxc2
                predicted_class(i) = 1;
            else
                predicted_class(i) = 2;
            end
        end
        

        error = (predicted_class~=validation_data(:,val_n));
        error_rate = sum(error)/val_m;
        
        %format = "%f     %f      %f\n";
        s = s + sprintf("%f     %f      %f\n",prior, (1-prior),error_rate);
        if error_rate < min_error
            min_error = error_rate;
            best_pc1 = prior;
            %sprintf(max_accuracy);
        end
    end %prior loop
    
    pc1 = best_pc1;
    pc2 = 1- best_pc1;
    
    disp(s);
 end
