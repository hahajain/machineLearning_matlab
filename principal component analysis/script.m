    clear all
    clc
    training_data = load('optdigits_train.txt');
    test_data = load('optdigits_test.txt');
    
    [m,n]= size(training_data);
    [m_test,n_test] = size(test_data);
    
    [eigenvectors, eigenvalues]= myPCA(training_data ); 
    
    sum1 = zeros(size(eigenvalues));
    
    for k = 1:size(eigenvalues,1)
        if k == 1
            sum1(k) = eigenvalues(k);
        else
            sum1(k) = sum1(k-1) + eigenvalues(k);
            
        end
        
    end
    
    sum_norm = sum1/sum(sum1,1);
    
    x_axis = linspace(1,64,64);
    plot(x_axis, sum_norm); 
    xlabel('Number of principal components');
    ylabel('Proportion of variance');
    
    sum2= 0;
    for i = 1:size(sum_norm,1)
        if sum2>=0.9
            break;
        else
            sum2 = sum2 + eigenvalues(i,1)/sum(eigenvalues);
        end
    end
    
    i-1;
    disp("Minimum number of eigenvectors that explain 90% variability is given by: ");
    disp(i-1);
       
   u = mean(training_data(:,1:n-1));
   
   proj_train =[(training_data(:,1:n-1)- u)*eigenvectors(:,1:i-1) training_data(:,n)];
   proj_test =[(test_data(:,1:n_test-1)- u)*eigenvectors(:,1:i-1) test_data(:,n_test)];
   
   
    for k = 1:2:7
        
        error = myKNN(proj_train, proj_test, k);
        disp(sprintf("KNN Error for k = %d",k));
        disp(error);
        
    end
  
