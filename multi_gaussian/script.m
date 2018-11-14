clc
clear all

training_data = load('training_data.txt');
test_data = load('test_data.txt');

for i = 1:3
    MultiGaussian(training_data , test_data, i);
end
