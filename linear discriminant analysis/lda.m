function[projection_matrix, eigenvectors_sorted, evalues_sorted] = myLDA(data, L)    
    
    [m,n] = size(data);
    
    C = unique(data(:,n));
    no_of_class = size(C,1);
    
    meani = zeros(size(C,1),n-1);
    size_class = zeros(no_of_class,1);
    
    for i = 1:no_of_class
        meani(i,:) = mean(data(data(:,n)==C(i),1:n-1));
        size_class(i) = size(data(data(:,n)==C(i)),1);
        
    end
    
    m_sum=sum(meani)/no_of_class;
    
    sb = zeros(n-1);
    
    for i = 1:no_of_class
        sb = sb + size_class(i)*(meani(i,:)-m_sum)'*(meani(i,:)-m_sum);
    end
    
    sw = zeros(n-1);
    
    for i = 1:no_of_class
            sw = sw + (data(data(:,n)==C(i),1:n-1)-meani(i,:))'*(data(data(:,n)==C(i),1:n-1)-meani(i,:));
    end
    
    S=pinv(sw)*sb;
    
    [V,D] = eig(S);
    
    eigenvalues = zeros(size(S,2),1);
    eigenvectors = zeros(size(V));
    for k = size(S,2):-1:1
        
        eigenvalues(size(S,2) - k +1) =  D(k,k);
        eigenvectors(:,size(S,2) - k +1) = V(:,k);
        
    end
    
    [evalues_sorted, indices] = sort(eigenvalues,'descend');
    eigenvectors_sorted = eigenvectors(:, indices);
    projection_matrix = data(:,1:n-1)*eigenvectors_sorted(:,1:L);
    eigenvectors_sorted=eigenvectors_sorted(:,1:L);
    
