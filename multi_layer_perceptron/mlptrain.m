function [zv, w, v, error_train, error_valid] = mlptrain(train_path, valid_path, k, l)

    data=load(train_path);
    data_valid=load(valid_path);
    [m,n]=size(data);
    [mv,nv]=size(data_valid);
    train_data = [ones(m,1) data(:,1:n-1)];
    valid_data = [ones(mv,1) data_valid(:,1:nv-1)];

    output=categorical(data(:,n));
    r=dummyvar(output);
    r_output=categorical(data_valid(:,nv));
    r_valid=dummyvar(r_output);

    w=-0.01+ (0.02)*rand(k,n);
    v=-0.01+ (0.02)*rand(l,k+1);

    step_size= 1e-5;

    curr_err=0;
    conv=0;
    itr=1;

    zv = zeros(k, m);

    while conv~=1
    
        prev_err = curr_err;
        curr_err=0;
        err=zeros(1,m);
    
        for t=1:m
        
            a=w*train_data(t,:)'; 
            a=max(a,0);            
        
            z = [1;a];            
        
            o=v*z;                 
        
            oexp = exp(o);
            y = oexp./sum(oexp);
        
            %compute dv and dw
            dv=step_size*(r(t,:)'-y)*z';  
        
            dw=step_size*(v(:,2:k+1)'*(r(t,:)'-y))*train_data(t,:);        
        
            dw(w<0)=0;
        
        %Update v and w
            v=v+dv;
            w=w+dw;
        
        %Error
            err(t) = r(t,:)*log(y);
        
            curr_err = curr_err + err(t);
            zv(:,t)=z(2:k+1,:);
        end
    
        dif = curr_err-prev_err;
    
        if abs(dif) < 0.01 || itr>1000
            conv = 1;
        end
    
        itr =itr+1;
    
        if itr>750
            step_size=1e-7;
        end
    
    end

    error =0;
    
%Classify train data
    for t=1:m
        a=w*train_data(t,:)';  
        a=max(a,0);            
        z = [1;a];            
        o=v*z;                
    
        oexp = exp(o);
        y = oexp./sum(oexp);
    
        [~,ind]=max(y);
    
        if r(t,ind)~=1
            error=error+1;
        end
    
    end
    error_train = error/m


%Classify valid data
    error=0;
    for t=1:m
        a=w*valid_data(t,:)';
        a=max(a,0);
        z = [1; a]; 
        o=v*z;
        oexp = exp(o);
        y= oexp./sum(oexp);
    
        [~,ind]=max(y);
    
        if r_valid(t,ind)~=1
            error=error+1;
        end
    
    end
    error_valid = error/mv
    
end
