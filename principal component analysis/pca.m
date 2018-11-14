function[eigenvectors, eigenvalues] = myPCA(training_data)    
    
    [m,n] = size(training_data);
    
    train_X = training_data(:,1:n-1);
    
    S = cov(train_X);
    [V,D] = eig(S);
    
    eigenvalues = zeros(size(S,2),1);
    eigenvectors = zeros(size(V));
    for k = size(S,2):-1:1
        
        eigenvalues(size(S,2) - k +1) =  D(k,k);
        eigenvectors(:,size(S,2) - k +1) = V(:,k);
        
    end   
