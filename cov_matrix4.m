% This is going to build up the covariance matrix
% a and b are vectors of variables
% h1 and h2 are the seperate output scales
% c is the time value of the change point

% An unscaled version of cov_matrix4

function  K = cov_matrix4(a,b,lambda,h1,h2,alpha,c,xc)

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
% we're fucked

    for i = 1:size_a
        for j = 1:size_b
            K(i,j) = k(a(i),b(j));
        end 
    end
end

% Lets atually do changepoint first

Ku = [K(1:xc,1:xc),sqrt(c)*K(1:xc,xc+1:end);...
    sqrt(c)*K(xc+1:end,1:xc),c*K(xc+1:end,xc+1:end)];

% Now the correlation between direct and indirect stuff.

K = [h1^2*Ku,alpha*h1*h2*Ku;alpha*h1*h2*Ku,h2^2*Ku];

end
