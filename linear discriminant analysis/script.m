
    clear all
    clc
    training_data = load('optdigits_train.txt');
    test_data = load('optdigits_test.txt');
    
    [m,n]= size(training_data);
    [m_test,n_test] = size(test_data);
    
    L = [2,4,9];
    
    for i = 1:size(L,2)
        [projected_matrix,eigenvectors, eigenvalues]= myLDA(training_data, L(i)); 
        projected_matrix_test = test_data(:,1:n_test-1)* eigenvectors;
        
        proj_train =[projected_matrix training_data(:,n)];
        proj_test =[projected_matrix_test test_data(:,n_test)];
   
        
        for k = 1:2:7
            
            error = myKNN(proj_train, proj_test, k);
            disp(sprintf("Error for L = %d and k = %d",L(i),k));
            disp(error)
        
        end
    end
    
