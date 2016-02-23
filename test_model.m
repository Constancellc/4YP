function [direct, indirect] = test_model

days = 1596;

t = 1:days;

rho_1 = 0.6;
rho_2 = 0.9;
cf = 3;
xc = 800;
x_l = 100;                  
x_s1 = 0.001;             
x_s2 = 0.002;              
a = 0.6;              

mu_d = 2.0;
mu_i = 1.8;

mu = [mu_d*ones(days,1);mu_i*ones(days,1)];

noise = [x_s1*eye(days,days),zeros(days,days);...
    zeros(days,days),x_s2*eye(days,days)];
        
cov = cov_matrix4(t,t,x_l,rho_1,rho_2,a,cf,xc) + noise;
%{
covk = cov_matrix2(t,t,x_l);

cov = [(rho_1^2)*covk,a*rho_1*rho_2*covk;...
    a*rho_1*rho_2*covk,(rho_2^2)*covk];
cov = cov+noise;
%}
x = mvnrnd(mu,cov);

direct = poissrnd(exp(x(1:days)));
indirect = poissrnd(exp(x(days+1:end)));
        
end
