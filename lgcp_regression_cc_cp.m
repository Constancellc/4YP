function [counts,variances,t] = lgcp_regression_cc

% Importing the no. of deaths / day data
direct_deaths = csvread('Data/Direct Frequencies.csv');
indirect_deaths = csvread('Data/Indirect Frequencies.csv');

% Vectors containing the numbers of days and weeks of the data
days = transpose([1:length(direct_deaths)]);

% Sampling the data to obtain a smaller number of points 'newlength' is the
% number of sample points you want to obtain. Max = 731
newlength = 70;

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

%options = optimoptions('fminunc','Algorithm','trust-region','GradObj','on','DerivativeCheck','on');

% Turning on gradient search and reducing the tolerance for the first round
% of 'rough' optiisations
options = optimoptions('fminunc','Algorithm','trust-region','GradObj','on','TolX',100);

% This is to suppress the output of the optimizer. It's irritating.
%options.Display = 'off';

% This is a function that limits its output between -1 and 1, it will be
% used to constrain the correlation parameter.
logit = @(x) (2./(1+exp(-x)))-1;
ilogit = @(x) -log((2./(x+1))-1);          % Inverse of function
dlogit = @(x) (1-x.^2)./(2);               % Derivative of function

% This section defines the starting points for the first 'rough' round of
% optimisation.

num = 20;                                   % Number of start points
precision = 10^-4;                          % Precision of start points
steps = 1/precision;

% Constraining the locations of the selected start points
h_upper = 3;                                % Maximum height start point
l_lower = 40;                               % Lowest length start point
l_upper = 100;                              % Maximum lengthstart point
n_upper = 0.01;                             % Maximum noise start point

% Compiling vectors of possible start points
h = [h_upper/steps:h_upper/steps:h_upper];  % Height
l_step = (l_upper-l_lower)/steps;         
l = [l_lower+l_step:l_step:l_upper];        % Length
n = [n_upper/steps:n_upper/steps:n_upper];  % Noise
c = [-1:2/steps:1];
c = c(2:end);                               % Correlation

% Setting up the latin hyper-cube, containing numbers from 0 to the #steps
X = steps*lhsdesign(num,6);

best = 10000;

% Rounding all entries to integer numbers
for i = 1:num
    for j = 1:6
        X(i,j) = round(X(i,j));
        if X(i,j) == 0
            X(i,j) = 1;
        end
    end
    
    % Fill array 'samples' with values from the predefined vectors for each
    % parameter
    samples(1) = log(h(X(i,1)));
    samples(2) = log(h(X(i,2)));
    samples(3) = l(X(i,3));
    samples(4) = log(n(X(i,4)));
    samples(5) = log(n(X(i,5)));
    samples(6) = ilogit(c(X(i,6)));

   
    try [x,fval] = fminunc(@product,[transpose(samples);logy(:,1);...
            logy(:,2)],options);
    catch 
        n1 = exp(samples(4));
        n2 = exp(samples(5));
        
        counter = 1;
        epsilon = 1e-6;
        
        while counter <=1
            samples(4) = log(n1 + epsilon);
            samples(5) = log(n2 + epsilon);
            
            try [x,fval] = fminunc(@product,[transpose(samples);logy(:,1);...
                    logy(:,2)],options);
            catch
                epsilon = epsilon*1.1
                counter = counter - 1;
            end
            
            counter = counter + 1;
        end
    end
    
    fval
    
    % If the function minimum is smaller than the previous best recorded
    if fval < best
        best = fval;                      % Replace best        
        x_ = x;
    end

end

%%%%%%%

x=x_;

rho1_final = exp(x(1));              % Replace hyper-parameters
rho2_final = exp(x(2));
l_final = (x(3));
s_final1 = exp(x(4));
s_final2 = exp(x(5));
a_final = logit(x(6));

% The following section is needed when we want to plot the output of the
% initial round of optimisation
%{
vi = x(7:end);
vi_final = [vi(1:newlength),vi(newlength+1:end)];

vari = hessian_diag(vi);

% Calculating the variance terms
variancei = [vari(1:newlength),vari(newlength+1:end)];

% Working out vectors of the mean +/- 2 std 
mean_plusi = zeros(2,newlength);
mean_minusi = zeros(2,newlength);

for k = 1:newlength
    for h = 1:2
        mean_plusi(h,k) = exp(vi_final(k,h) + 2*sqrt(variancei(k,h)));
        mean_minusi(h,k) = exp(vi_final(k,h) - 2*sqrt(variancei(k,h)));
    end
end

Y1i = [mean_minusi(1,:),fliplr(mean_plusi(1,:))];
Y2i = [mean_minusi(2,:),fliplr(mean_plusi(2,:))];

%}

%%%%%%%

best

optionsnew = optimoptions('fminunc','Algorithm','trust-region','GradObj','on','TolX',1e-4);
%optionsnew = optimoptions('fminunc','Algorithm','trust-region','GradObj','on');
[x,fval] = fminunc(@product,x_,optionsnew);

rho1_final = exp(x(1));              % Replace hyper-parameters
rho2_final = exp(x(2));
l_final = (x(3));
s_final1 = exp(x(4));
s_final2 = exp(x(5));
a_final = logit(x(6));

v_ = x(7:end);                    % Replace poisson rates

% Split into seperate data streams
v_final = [v_(1:newlength),v_(newlength+1:end)];

fval

var = hessian_diag(v_);

% Calculating the variance terms
variance = [var(1:newlength),var(newlength+1:end)];

% For the testing script
variances = variance;

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


% Remove Anomili
indirect_deaths(1328) = 0;
direct_deaths(1328) = 0;

figure                                % Create a new figure

% This is the first subplot if we want the output of the initial round
%{
subplot(2,1,1)
shade = fill(X,Y1i,'b');               % Fills the confidence region   
set(shade,'facealpha',.2)             % Sets the region transparent
set(shade,'EdgeColor','None')
hold on
shade2 = fill(X,Y2i,'g');              
set(shade2,'facealpha',.2)           
set(shade2,'EdgeColor','None')
plot(t,exp(vi_final))             % Plots the predicted mean values
hold on
plot(t,y,'x');                   % Plots the training data

subplot(2,1,2)
%}

% This is if we want to plot the test and training data
subplot(2,1,1)
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
title('Incidents - Training Data')
xlabel('Time /Days')
ylabel('Incidents')

subplot(2,1,2)
plot(days,direct_deaths,'b');
hold on
plot(days,indirect_deaths,'g');
hold on
shade = fill(X,Y1,'b');               % Fills the confidence region   
set(shade,'facealpha',.2)             % Sets the region transparent
set(shade,'EdgeColor','None')
hold on
shade2 = fill(X,Y2,'g');              
set(shade2,'facealpha',.2)           
set(shade2,'EdgeColor','None')
plot(t,exp(v_final))             % Plots the predicted mean values

title('Incidents - Full Testing Data')
xlabel('Time /Days')
ylabel('Incidents')
%}
%-------------------------------
% END OF MODEL PLOTTING SECTION
%-------------------------------



%-------------------------------------------------------
% START OF LOCAL FUNCTION SECTION
%-------------------------------------------------------


% Function which calculates selected terms of the posterior numerator.
    function [f,g] = product(x)
        rho_1a = exp(x(1));
        rho_1b = exp(x(2));
        rho_2a = exp(x(3));
        rho_2b = exp(x(4));
        xc = x(5);                     % Index of changepoint
        x_l = (x(6));                  % Length scale
        x_s1 = exp(x(7));              % Direct noise variance
        x_s2 = exp(x(8));              % Indirect noise variance
        a = logit(x(9));               % Correlation factor
        
        v = x(10:end);                  % Predicted poisson rates
        
        % Finding covariance matrix
        cov11 = cov_matrix3(t,t,x_l,rho_1a^2,rho_1b^2,xc);
        cov12 = cov_matrix3(t,t,x_l,rho_1a*rho2_a,rho_1b*rho_2b,xc);
        cov22 = cov_matrix3(t,t,x_l,rho_2a^2,rho_2b^2,xc);
        
        noise1 = x_s1*eye(newlength,newlength);
        noise2 = x_s2*eye(newlength,newlength);

        cov = [cov11+noise1,a*cov12;a*cov12,cov22+noise2];
        
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
        
        % For the purpose of the derivatives it's useful to work out the
        % matrix without any output scale
        
        K = cov_matrix2(t,t,x_l);
        
        % Ok, derivatives are going to be a bitch
        
        % Starting with the dervative of the covariance w.r.t rho_1a
        
        covdr1a_11 = [2*rho_1a*K(1:xc,1:xc),rho_1b*K(1:xc,xc+1:end);...
            rho_1b*K(xc+1:end,1:xc),zeros((newlength-xc),(newlength-xc))];
        covdr1a_12 = a*[rho_2a*K(1:xc,1:xc),...
            0.5*sqrt((rho_1b*rho_2a*rho_2b)/rho_1a)*K(1:xc,xc+1:end);...
            0.5*sqrt((rho_1b*rho_2a*rho_2b)/rho_1a)*K(xc+1:end,1:xc),...
            zeros((newlength-xc),(newlength-xc))];
        
        covdr1a = [covdr1a_11,covdr1a_12;covdr1a_12;zeros(newlength,newlength)];
        
        % Now doing the same for rho_1b
        
        covdr1b_11 = [zeros(xc,xc),rho_1a*K(1:xc,xc+1:end);...
            rho_1a*K(xc+1:end,1:xc),2*rho_1b*K(xc+1:end,xc+1:end)];
        covdr1b_12 = a*[zeros(xc,xc),...
            0.5*sqrt((rho_1a*rho_2a*rho_2b)/rho_1b)*K(1:xc,xc+1:end);...
            0.5*sqrt((rho_1a*rho_2a*rho_2b)/rho_1b)*K(xc+1:end,1:xc),...
            rho_2b*K(xc+1:end,xc+1:end)];
        
        covdr1b = [covdr1b_11,covdr1b_12;covdr1b_12;zeros(newlength,newlength)];
        
        % Now the gradients for the output scales rho1 and rho2
        covdr1 = [2*rho_1*covk,a*rho_2*covk;a*rho_2*covk,zeros(newlength,newlength)];
        covdr2 = [zeros(newlength,newlength),a*rho_1*covk;a*rho_1*covk,2*rho_2*covk];
        
        gr1 = 0.5*rho_1*(-transpose(d)*(R\(L\covdr1))*(R\(L\d))+trace(R\(L\covdr1)));
        gr2 = 0.5*rho_2*(-transpose(d)*(R\(L\covdr2))*(R\(L\d))+trace(R\(L\covdr2)));
                
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
        
        cov_ = [(rho1_final^2)*covk+noise1,a_final*rho1_final*rho2_final*covk;...
            a_final*rho1_final*rho2_final*covk,(rho2_final^2)*covk+noise2];
        
        % Cholesky decomposition to avoid inverting covariance
        L = chol(cov_,'lower');
        R = transpose(L);
        
        icov = R\(L\eye(2*newlength));
        
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

end
