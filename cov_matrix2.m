% This is going to build up the covariance matrix
% a and b are vectors of variables

function K = cov_matrix2(a,b,lambda)

size_a = length(a);
size_b = length(b);

% Initialising K as an empty matrix of the right size.
K = zeros(size_a,size_b);

% Defining the covariance function
% ai and bj are elements of the input vectors

% Using the squared exponential function.
k = @(ai,bj) (exp((-1/2).*((ai-bj)./lambda).^2));

% Avoiding unecessary calculations in the case that the two input vectors
% are the same.

if size_a == size_b
    K = bsxfun(k,a,transpose(b));
    
else


    for i = 1:size_a
        for j = 1:size_b
            K(i,j) = k(a(i),b(j));
        end 
    end
end
%K = K + noise*eye(size(K));
end
