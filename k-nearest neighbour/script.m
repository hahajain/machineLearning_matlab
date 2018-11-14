    clear all
    clc
    training_data = load('optdigits_train.txt');
    test_data = load('optdigits_test.txt');
    
    for k = 1:2:7
        
        error = myKNN(training_data, test_data, k);
        disp(sprintf("KNN Error for k = %d",k));
        disp(error);
    end
