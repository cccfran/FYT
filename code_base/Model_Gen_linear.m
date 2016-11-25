function [ X, y, true_w ] = Model_Gen_linear( para )

%   Model: y = Xw + e
%   1. X: [m*n], each column of X is one sample data;
%   2. y: [n*1], is the label of each sample data.A(i,:).
%   3. m: [m*1], is the number of features.
rng(23)

para_noise  = para.noise;
n           = para.size_of_data;
m           = para.size_of_features;

true_w      = randi([-10 10], m, 1);
X           = randn(m, n);
noise       = para_noise * randn(n, 1);
y           = X' * true_w + noise;

end

