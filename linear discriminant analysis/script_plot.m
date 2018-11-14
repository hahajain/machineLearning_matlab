    clear all
    clc
    training_data = load('optdigits_train.txt');
    test_data = load('optdigits_test.txt');
    
    [m,n]= size(training_data);
    [m_test,n_test] = size(test_data);
  
    [projected_matrix,eigenvectors, eigenvalues]= myLDA(training_data, 2); 
    [projected_matrix_test,eigenvectors_test, eigenvalues_test]= myLDA(test_data, 2);
        
     proj_train =[projected_matrix training_data(:,n)];
     proj_test =[projected_matrix_test test_data(:,n_test)];
   
    dx = 0.1; dy = 0.1;
    figure
    subplot(2,1,1);
    scatter(proj_train(:,1),proj_train(:,2),20,proj_train(:,3),'filled');
    title('Training Data');
    for i = 0:size(unique(training_data(:,n)),1)-1
        classi = proj_train(proj_train(:,3)==i,1:2);
        rand_index = randsample(1:length(classi),6);
        cprint = classi(rand_index,:);
        text(cprint(:,1)+dx, cprint(:,2)+dy, string(i));
    end
   
    subplot(2,1,2);
    scatter(proj_test(:,1),proj_test(:,2),20,proj_test(:,3),'filled')
    title('Test Data');
    for i = 0:size(unique(test_data(:,n)),1)-1
        classi = proj_test(proj_test(:,3)==i,1:2);
        rand_index = randsample(1:length(classi),6);
        cprint = classi(rand_index,:);
        text(cprint(:,1)+dx, cprint(:,2)+dy, string(i));
    end
  
