% This is going to build up the covariance matrix
% a and b are vectors of variables

function K = cov_matrix(a,b,h,lambda)

size_a = length(a);
size_b = length(b);

% Initialising K as an empty matrix of the right size.
K = zeros(size_a,size_b);

% Defining the covariance function
% ai and bj are elements of the input vectors

%h = 10;                          % controls gain
%lambda = 3.2;                     % controls smoothness

% Using the squared exponential function.
k = @(ai,bj) h.^2*(exp((-1).*((ai-bj)./lambda).^2));

for i = 1:size_a
    for j = 1:size_b
        K(i,j) = k(a(i),b(j));
    end 
end
