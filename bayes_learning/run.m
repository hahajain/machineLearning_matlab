training_data = load('SPECT_train.txt');
validation_data = load('SPECT_valid.txt');
test_data = load('SPECT_test.txt');


% Bayes
[p1, p2, pc1, pc2] = Bayes_Learning(training_data , validation_data);
Bayes_Testing(test_data , p1, p2, pc1, pc2)
