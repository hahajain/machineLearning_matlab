function Bayes_Testing(test_data, p1, p2, pc1, pc2)

    [test_m, test_n] = size(test_data);
    predicted_class = zeros(test_m,1);
    error_rate = 0;
    
    for i = 1:test_m
        pxc1 = 1;
        pxc2 = 1;
            
        for j = 1:test_n - 1
            pxc1 = pxc1*power(p1(j),1-test_data(i,j))*power(1-p1(j),test_data(i,j));
            pxc2 = pxc2*power(p2(j),1-test_data(i,j))*power(1-p2(j),test_data(i,j));
        end
            
        if pc1*pxc1 > pc2*pxc2
            predicted_class(i) = 1;
        else
            predicted_class(i) = 2;
        end
        
        error = (predicted_class~=test_data(:,test_n));
        error_rate = sum(error)/test_m;
    end
    
    error_rate
end
