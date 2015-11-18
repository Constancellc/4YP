function [counts,mean_plus,mean_minus,t] = lgcp_regression

% Importing the no. of deaths / day data
direct_deaths = csvread('Data/Direct Frequencies.csv');
indirect_deaths = csvread('Data/Indirect Frequencies.csv');

% Vectors containing the numbers of days and weeks of the data
days = transpose([1:length(direct_deaths)]);

% Sampling the data to obtain a smaller number of points 'newlength' is the
% number of sample points you want to obtain. Max = 731
newlength = 100;

% Setting out the vectors to be filled by the sampled points
direct_sampled = zeros(newlength,1);
indirect_sampled = zeros(newlength,1);

% Working out the necessary sampling frequency
inte = floor(length(direct_deaths)/newlength);

% Filling the sampled vector
for i = 1:newlength
    direct_sampled(i) = direct_deaths(i*inte);
    indirect_sampled(i) = indirect_deaths(i*inte);
end
    
% A vector of the number of days into conflict for plot
t = [inte:inte:length(direct_deaths)];
if length(t) >= newlength + 1
    t = t(1:newlength);
end

y = [direct_sampled,indirect_sampled];

% Working out the average log(y) values, for use as the GP means
sampled_direct_mean = 0;
sampled_indirect_mean = 0;

% Work out the log of all of the observed counts
logy = zeros(newlength,2);

for k = 1:2
    for j = 1:newlength
        % Don't try and calculate log(0)
        if y(j,k) ~= 0
            logy(j,k)=log(y(j,k));
        end
    end
end

% Find the log-average for each stream
mean = [sum(logy(:,1)),sum(logy(:,2))]/newlength;

% Turning on gradient search for the poisson rates
options = optimoptions('fminunc','Algorithm','trust-region','GradObj','on');

% This is to suppress the output of the optimizer. It's irritating.
options.Display = 'off';
options2 = optimset('Display', 'off');

% Set up the initial guess for the hyperparameters
h = [3;3];
l = [100;85];

for q = 1:2
    
    % Setting the hyper parameters to the starting point
    test_h = h(q); test_l = l(q); test_s = 0.001;

    % Learning the optimum values for the poisson rates
    v_test = fminunc(@product,logy(:,q),options);

    % Updating our guess of the hyper-parameters
    [x,fval] = fmincon(@hyper,[test_h;test_l;test_s],[],[],[],[],[0;0;0],[100;400;1],[],options2);
	test_h = x(1); test_l = x(2); test_s = x(3);

    % Updating our guess for the hyper-parameters
    v_final(:,q) = fminunc(@product,v_test,options);
    
    % Saving the final hyper-parameters
    h_final(q) = x(1); l_final(q) = x(2); s_final(q) = x(3);
    
    % Calculating the variance terms
    variance(:,q) = hessian_diag(v_final);

end   

% The vector of predicted numbers of incidents
counts = exp(v_final);

% Working out vectors of the mean +/- 2 std 
mean_plus = zeros(2,newlength);
mean_minus = zeros(2,newlength);

for k = 1:newlength
    for h = 1:2
        mean_plus(h,k) = exp(v_final(k,h) + 2*sqrt(variance(k,h)));
        mean_minus(h,k) = exp(v_final(k,h) - 2*sqrt(variance(k,h)));
    end
end

%---------------------------------
% START OF MODEL PLOTTING SECTION
%---------------------------------
% {

% Setting up vectors for the shade function
X = [t,fliplr(t)];
Y1 = [mean_minus(1,:),fliplr(mean_plus(1,:))];
Y2 = [mean_minus(2,:),fliplr(mean_plus(2,:))];

figure                                % Create a new figure

subplot(2,1,1)                        % First the direct incidents
shade = fill(X,Y1,'r');               % Fills the confidence region   
set(shade,'facealpha',.2)             % Sets the region transparent
set(shade,'EdgeColor','None')
hold on
plot(t,exp(v_final(:,1)))             % Plots the predicted mean values
hold on
plot(t,y(:,1),'x');                   % Plots the training data

% Now the same but for the indirect data stream
subplot(2,1,2)
shade2 = fill(X,Y2,'r');              
set(shade2,'facealpha',.2)           
set(shade2,'EdgeColor','None')
hold on
plot(t,exp(v_final(:,2)))
hold on
plot(t,y(:,2),'x');

%}
%-------------------------------
% END OF MODEL PLOTTING SECTION
%-------------------------------

%-------------------------------------------------------
% START OF HYPER-PARAMETERS LIKELIHOOD PLOTTING SECTION
%-------------------------------------------------------
%{

% Setting up the vectors of hyperparameters to be plotted
h = [0.01:0.01:20];
h_test = zeros(length(h),2);

l = [0.1:0.1:500];
l_test = zeros(length(l),2);

n = [0.000001:0.000001:0.0001];
n_test = zeros(length(n),2);

% Working out the likelihood for each test parameter
for q = 1:2;
    v_test = v_final(:,q);

    for i = 1:length(l)
        l_test(i,q) = exp(-hyper([h_final(q);l(i);s_final(q)]));
    end

    for i = 1:length(h)
        h_test(i,q) = exp(-hyper([h(i);l_final(q);s_final(q)]));
    end

    for i = 1:length(n)
        n_test(i,q) = exp(-hyper([h_final(q);l_final(q);n(i)]));
    end
end

figure                             % Plotting the variation for all three
subplot(2,2,1)
semilogy(h,h_test)

subplot(2,2,2)
semilogy(l,l_test)

subplot(2,2,3)
semilogy(n,n_test)

%}
%-----------------------------------------------------
% END OF HYPER-PARAMETERS LIKELIHOOD PLOTTING SECTION
%-----------------------------------------------------

% Function which calculates selected terms of the posterior numerator. It
% takes the values for poisson rate as inputs and uses globally defined
% hyper-parameters and a global variable q to indicate which data stream.
    function [f,g] = product(v)
        
        cov = cov_matrix(t,t,test_h,test_l) +... 
            (test_s+10^-9)*eye(newlength);
        
        icov = cov\eye(size(cov));
        
        d = v-mean(q)*ones(newlength,1);
        
        % Containing only the terms which change with v
        f = sum(exp(v))-transpose(v)*y(:,q)+0.5*log(det(cov))...
            +0.5*transpose(d)*(cov\d);

        % Setting up a vector of the gradient values
        g = zeros(newlength,1);
        
        for gi = 1:newlength
            gterm = 0;
            
            for gk = 1:newlength;
                gterm = gterm + (v(gk)-mean(q))*(icov(gk,gi)+icov(gi,gk));
            end
            
            g(gi) = exp(v(gi))-y(gi,q) + 0.5*gterm;
        end
            
    end

% Function which calculates selected terms from the likelihood for a
% globally defined set of poisson rate for inputted hyper-parameters
    function f = hyper(x)
        
        cov = cov_matrix(t,t,x(1),x(2)) + x(3)*eye(newlength);
        
        d = v_test-mean(q)*ones(newlength,1);

        f = 0.5*log(det(cov))+0.5*transpose(d)*(cov\d);
       
    end

% Function which calculates the diagonal terms of the hessian matrix given
% a set of intensities, and using global hper-parameters. This vecotor will
% be the predicted variances for each timestep. 
    function va = hessian_diag(v)
        
        cov_ = cov_matrix(t,t,h_final(q),l_final(q));
        cov_ = cov_ + s_final(q)*eye(newlength);
        
        icov = cov_\eye(size(cov_));
        
        A = eye(newlength);
        
        for ai = 1:newlength
            for aj = 1:newlength
                if ai == aj
                    A(ai,aj) = exp(v(ai)) + icov(ai,aj);
                else
                    A(ai,aj) = 0.5*(icov(ai,aj) + icov(aj,ai));
                end
            end
        end
        
        hessian = inv(A);
        
        va = diag(hessian);
    end

end

