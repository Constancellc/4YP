function [counts,mean_plus,mean_minus,t] = lgcp_regression_cc

% Importing the no. of deaths / day data
direct_deaths = csvread('Data/Direct Frequencies.csv');
indirect_deaths = csvread('Data/Indirect Frequencies.csv');

% Vectors containing the numbers of days and weeks of the data
days = transpose([1:length(direct_deaths)]);

% Sampling the data to obtain a smaller number of points 'newlength' is the
% number of sample points you want to obtain. Max = 731
newlength = 80;

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

% These are matricies which will be used for a plot of both data streams.
x2 = [t;t];                                 % Contains the sample times
y = [direct_sampled,indirect_sampled];      % Contains the sample incidents

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

% Turning on gradient search for the poisson rates, the tolerance is
% reduced in order to speed up optimisation.
%options = optimoptions('fminunc','Algorithm','trust-region','GradObj','on','DerivativeCheck','on');
options = optimoptions('fminunc','Algorithm','trust-region','GradObj','on','TolX',1e-2);

% This is to suppress the output of the optimizer. It's irritating.
%options.Display = 'off';

% This is a function that limits its output between -1 and 1, it will be
% used to constrain the correlation parameter.
logit = @(x) (2./(1+exp(-x)))-1;
ilogit = @(x) -log((2./(x+1))-1);            % Inverse of function
dlogit = @(x) (1-x.^2)./(2);                 % Derivative of function

% This is the number of starting points we are going to try for our
% optimisation.
num = 1;

% Chose a precision for the trial start points of the optimisation
precision = 10^-4;
steps = 1/precision;

% Define the possible start values for the output scales
h_upper = 2;                     % Maximum value
h = [h_upper/steps:h_upper/steps:h_upper];

% Define the possible start values for the input scale
l_lower = 20;                    % Lowest value
l_upper = 220;                   % Maximum value
l_step = (220-20)/steps;         
l = [l_lower+l_step:l_step:l_upper];

% Define the possible start values for the noise values
noise_upper = 0.01;
n = [noise_upper/steps:noise_upper/steps:noise_upper];

% Define the possible start values for the correlation parameter
c = [-1:2/steps:1];
c = c(2:end);

% Setting up the latin hyper-cube, containing numbers from 0 to the #steps
X = steps*lhsdesign(num,6);

% Rounding all entries to integer numbers
for i = 1:num
    for j = 1:6
        X(i,j) = round(X(i,j));
    end
    
    % Fill array 'samples' with values from the predefined vectors for each
    % parameter
    samples(i,1) = h(X(i,1));
    samples(i,2) = h(X(i,2));
    samples(i,3) = l(X(i,3));
    samples(i,4) = n(X(i,4));
    samples(i,5) = n(X(i,5));
    samples(i,6) = ilogit(c(X(i,6)));
end

best = 0;

for i = 1:num
    
    [x,fval] = fminunc(@product,[transpose(samples(num,:));logy(:,1);...
        logy(:,2)],options);
    
    % If the function minimum is smaller than the previous best recorded
    if fval < best
        best = fval;                      % Replace best
        
        rho1_final = (x(1));              % Replace hyper-parameters
        rho2_final = (x(2));
        l_final = (x(3));
        s_final1 = exp(x(4));
        s_final2 = exp(x(5));
        a_final = logit(x(6));

        v_ = x(7:end);                    % Replace poisson rates
        % Duplicated for both data streams
        v_final = [v_(1:newlength),v_(newlength+1:end)];
    end

end       


%[x,fval] = fminunc(@product,[0.5;0.6;70;log(0.0001);log(0.0001);0.8;logy(:,1);logy(:,2)],options);

%rho1_final = exp(x(1));
%rho2_final = exp(x(2));

var = hessian_diag(v_);

% Calculating the variance terms
variance = [var(1:newlength),var(newlength+1:end)];

l_final
rho1_final
rho2_final
s_final1
s_final2
a_final

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

shade = fill(X,Y1,'b');               % Fills the confidence region   
set(shade,'facealpha',.2)             % Sets the region transparent
set(shade,'EdgeColor','None')
hold on
shade2 = fill(X,Y2,'g');              
set(shade2,'facealpha',.2)           
set(shade2,'EdgeColor','None')
plot(t,exp(v_final))             % Plots the predicted mean values
hold on
plot(t,y,'x');                   % Plots the training data



%p = posterior(v_final(:,1))

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

% Function which calculates selected terms of the posterior numerator.
    function [f,g] = product(x)
        rho_1 = (x(1));                % Direct output scale
        rho_2 = (x(2));                % Indirect output scale
        %rho_1 = exp(x(1));
        %rho_2 = exp(x(2));
        x_l = (x(3));                  % Length scale
        x_s1 = exp(x(4));              % Direct noise variance
        x_s2 = exp(x(5));              % Indirect noise variance
        a = logit(x(6));               % Correlation factor
        
        v = x(7:end);                  % Predicted poisson rates
        
        % Finding covariance matrix
        covk = cov_matrix2(t,t,x_l);
        noise1 = x_s1*eye(newlength,newlength);
        noise2 = x_s2*eye(newlength,newlength);
                
        cov = [(rho_1^2)*covk+noise1,a*rho_1*rho_2*covk;...
            a*rho_1*rho_2*covk,(rho_2^2)*covk+noise2];
        
        % Cholesky decomposition to avoid inverting covariance
        L = chol(cov,'lower');
        R = transpose(L);

        % difference vector of the observed values
        d = v-[mean(1)*ones(newlength,1);mean(2)*ones(newlength,1)];
        
        % The quantity we are trying to minimize - the negatie of the log
        % likelihood containing only the terms which change with v
        f = sum(exp(v))-transpose(v)*[y(:,1);y(:,2)]+0.5*log(det(cov))...
            +0.5*transpose(d)*(cov\d);

        % Gradients for the poisson rate values
        g = R\(L\d)+exp(v)-[y(:,1);y(:,2)];
        
        % Now the gradients for the output scales rho1 and rho2
        covdr1 = [2*rho_1*covk,a*rho_2*covk;a*rho_2*covk,zeros(newlength,newlength)];
        covdr2 = [zeros(newlength,newlength),a*rho_1*covk;a*rho_1*covk,2*rho_2*covk];
        
        gr1 = 0.5*(-transpose(d)*(R\(L\covdr1))*(R\(L\d))+trace(R\(L\covdr1)));
        gr2 = 0.5*(-transpose(d)*(R\(L\covdr2))*(R\(L\d))+trace(R\(L\covdr2)));
                
        % Next the gradient for the length scale
        for ci = 1:newlength
            for cj = 1:newlength
                dcov(ci,cj) = (t(ci)-t(cj))^2*covk(ci,cj);
            end
        end

        dcov = [(rho_1^2)*dcov,a*rho_1*rho_2*dcov;...
            a*rho_1*rho_2*dcov,(rho_2^2)*dcov]/(x_l^3);
        
        gl = 0.5*(-transpose(d)*(R\(L\dcov))*(R\(L\d))+trace(R\(L\dcov)));
        
        % Now the gradients for the noise values
        dcovs1 = [eye(newlength),zeros(newlength,newlength);zeros(newlength,2*newlength)];
        dcovs2 = [zeros(newlength,2*newlength);zeros(newlength,newlength),eye(newlength)];
        
        gs1 = 0.5*x_s1*(-transpose(d)*(R\(L\dcovs1))*(R\(L\d))+trace(R\(L\dcovs1)));
        gs2 = 0.5*x_s2*(-transpose(d)*(R\(L\dcovs2))*(R\(L\d))+trace(R\(L\dcovs2)));
        
        % Last the gradient for the correlation factor
        covda = [zeros(newlength,newlength),rho_1*rho_2*covk;rho_2*rho_1*covk,zeros(newlength,newlength)];
        
        ga = 0.5*dlogit(a)*(-transpose(d)*(R\(L\covda))*(R\(L\d))+trace(R\(L\covda)));
        
        % Compiling the gradients into one vector
        g = [gr1;gr2;gl;gs1;gs2;ga;g];
    end

% Function which calculates the diagonal terms of the hessian matrix given
% a set of intensities, and using global hper-parameters. This vecotor will
% be the predicted variances for each timestep. 
    function va = hessian_diag(v)
        
        covk = cov_matrix2(t,t,l_final);
        noise1 = s_final1*eye(newlength,newlength);
        noise2 = s_final2*eye(newlength,newlength);
        
        %covk = covk+noise;
        
        %cov_ = [(rho1_final^2)*covk+noise,rho1_final*rho2_final*covk+noise;...
        %    rho1_final*rho2_final*covk+noise,(rho2_final^2)*covk+noise];
        
        %cov_ = [(rho1_final^2)*covk,rho1_final*rho2_final*covk;...
        %    rho1_final*rho2_final*covk,(rho2_final^2)*covk];
        
        cov_ = [(rho1_final^2)*covk+noise1,a_final*rho1_final*rho2_final*covk;...
            a_final*rho1_final*rho2_final*covk,(rho2_final^2)*covk+noise2];
        
        
        icov = cov_\eye(size(cov_));
        
        A = eye(2*newlength);
        
        for ai = 1:newlength*2
            for aj = 1:newlength*2
                if ai == aj
                    A(ai,aj) = exp(v(ai)) + icov(ai,aj);
                else
                    A(ai,aj) = 0.5*(icov(ai,aj) + icov(aj,ai));
                end
            end
        end
        
        covariance = inv(A);
        
        va = diag(covariance);
    end

%{
    function p = posterior(v)
        
        cov = cov_matrix(x2,x2,h_final(q),l_final(q)) +... 
            (s_final(q)+10^-9)*eye(newlength);
        
        icov = cov\eye(size(cov));
        
        d = v-mean(q)*ones(newlength,1);
        
        y_fact = 1;
        
        for yi = 1:newlength
            y_fact = y_fact/factorial(y(yi,q));
        end
        
        log_product = -sum(exp(v))+transpose(v)*y(:,q)+log(y_fact)-...
            (newlength/2)*log(2*pi)-0.5*log(det(cov))-0.5*transpose(d)*(cov\d);
        
        product = exp(log_product);
        
        v0 = v_final(:,q);
        d0 = v0-mean(q)*ones(newlength,1);
        
        log_p0 = -sum(exp(v0))+transpose(v0)*y(:,q)+log(y_fact)-...
            (newlength/2)*log(2*pi)-0.5*log(det(cov))-0.5*transpose(d0)*(cov\d0);
        p0 = exp(log_p0);
        
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
        
        factor = p0*sqrt(((2*pi)^newlength)/det(A));
        
        p = product / factor;
    end

%}
end



