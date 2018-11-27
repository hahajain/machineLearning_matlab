function [zv] = mlptest(test_path, w,v)

    data=load(test_path);
    
    
    [m,n]=size(data);
    
    test_data = [ones(m,1) data(:,1:n-1)];
    output=categorical(data(:,n));
    r=dummyvar(output);
    error=0;
    zv = zeros(18, m);
    for t=1:m
        a=w*test_data(t,:)';  
        a=max(a,0);            
        z = [1;a];            
        o=v*z;                
    
        oexp = exp(o);
        y = oexp./sum(oexp);
    
        [~,ind]=max(y);
        
        if r(t,ind)~=1
            error=error+1;
        end
        
    
        zv(:,t)=z(2:end);
        
    
    end
    error_test = error/m
    fprintf("Testing error rate is %d for 18 no. of hidden units \n",error_test);
 
end
