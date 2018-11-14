function [gamma, u, llm] = EMG(flag, imag, k)
    

    [img,cmap] = imread(imag);
    img_rgb = ind2rgb(img,cmap);
    img_double = im2double(img_rgb);
    data = reshape(img_double,[],3);
    [m,n] = size(data);
    lambda = 0.00001;
    
    
     sigma = zeros(n,n,k);
     prior = zeros(k,1);
     
     [idx,u] = kmeans(data,k,'MaxIter',3,'EmptyAction','singleton');
     for i = 1:k
         if flag== 1
             sigma(:,:,i) = cov(data(idx==i,:)) + lambda*eye(3,3);
         else
             sigma(:,:,i) = cov(data(idx==i,:));
         end
        
         prior(i) = sum(idx(:)==i)/m;
     end
      
     norm = zeros(m,k);
     lle = zeros(100,1);
     llm = zeros(100,1);
     mvpd = zeros(m,k);
     
          
     for iter = 1:100
         
         disp(iter);
        
        try
            for j = 1:k
                    mvpd(:,j) = mvnpdf(data, u(j,:), sigma(:,:,j));
                    norm(:,j)  = prior(j) * mvpd(:,j);
            end
        catch
            error('Sigma is not a positive-definite. Try again');
        end
        
        sum_clust = sum(norm,2);
        gamma = norm./sum_clust;
        
        %complete log liklihood after e step
        
        mvpd(mvpd==0)=eps;
        prior(prior==0)=eps;
                      
        try
            for i=1:m
                for j =1:k
                    lle(iter) = lle(iter) + gamma(i,j)*(log(prior(j))+log(mvpd(i,j)));
                end
            end
        catch
           error('Sigma is not a positive-definite. Try again');
        end
        
        newu= gamma'*data;
        ni = sum(gamma,1)';
        
        updated_u = newu./ni;
        updated_prior = ni/m;
        
        updated_sigma = zeros(n,n,k);
        %update sigma
        
        for i = 1:k
            for j = 1:m
                if flag == 0
                    updated_sigma(:,:,i) = updated_sigma(:,:,i) + gamma(j,i).*(data(j,:)-u(i,:))'*(data(j,:)-u(i,:));
                else
                    updated_sigma(:,:,i) = updated_sigma(:,:,i) + gamma(j,i).*(data(j,:)-u(i,:))'*(data(j,:)-u(i,:)) + lambda*eye(3,3);
                end
            end
            updated_sigma(:,:,i)=updated_sigma(:,:,i)./ni(i);
        end
      
        
        %complete log liklihood after m step
        
        sigma = updated_sigma;
        u= updated_u;
        prior = updated_prior;
        
        try
            for j = 1:k
                    mvpd(:,j) = mvnpdf(data, u(j,:), sigma(:,:,j));
            end
        catch
            error('Sigma is not a positive-definite. Try again');
        end
       
        mvpd(mvpd==0)=eps;
        prior(prior==0)=eps;
     
        try
            for i=1:m
                for j =1:k
                    llm(iter) = llm(iter) + gamma(i,j)*(log(updated_prior(j))+log(mvpd(i,j)));
                end
            end
        catch
           error('Sigma is not a positive-definite. Try again');
        end
        
     end
   
     cluster_indexes = zeros(n,1);
     for i = 1:m
        [val, idxx] = max(gamma(i,:));
        cluster_indexes(i) = idxx;
     end
     
     rgb_data = zeros(m,3);
     for i = 1:m
        rgb_data(i,:) = u(cluster_indexes(i),:);
     end
     
     compress_img = reshape(rgb_data,size(img_rgb,1),size(img_rgb,2),3);
     figure
     imagesc(compress_img);
     
     figure
     x = linspace(1,iter,iter);
     scatter(x,llm(1:iter),'filled');
     hold on;
     y = linspace(1,iter,iter);
     scatter(y,lle(1:iter),'filled');
      
end
