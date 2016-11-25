function [ X, y, true_w ] = Model_Gen_logistics( para )
% 	To be Finished !!!
%   Model: y =  1 / (1 + e^(-Xw))
%   1. X: [n*m], each column of X is one sample data;
%   2. y: [n*1], is the label of each sample data.A(i,:).

para_noise  = para.noise;
n           = para.size_of_data;
m           = para.size_of_features;

true_w      = randi([-10 10], m, 1);
X           = randn(n, m);
noise       = para_noise * randn(n, 1);
y           = 1 / (1 + exp(-X*true_w+noise));


end

