function l = likelihood(cov,y,mean)
% This functuon computes the likelihood of a model with covariance 'cov'
% and mean function 'mean' being the underlying system for training data
% contained in y

mag = norm(cov,2);
p = length(cov);   % Same as no. variables

if mag <= 0
    l = 0;
else
    d = y-mean*ones(p,1);
    l = (1/(((2*pi)^(1/2))*sqrt(mag)))*exp(-0.5*transpose(d)*inv(cov)*d);
end
%l = log(l);
end
