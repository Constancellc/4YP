function [counts,variances,t] = lgcp_regression_ccp

% {
% Importing the no. of deaths / day data
direct_deaths = csvread('Data/Direct Frequencies.csv');
indirect_deaths = csvread('Data/Indirect Frequencies.csv');
%}

%[direct_deaths, indirect_deaths] = test_model();

% Vectors containing the numbers of days and weeks of the data
days = transpose([1:length(direct_deaths)]);

% Sampling the data to obtain a smaller number of points 'newlength' is the
% number of sample points you want to obtain. Max = 731
newlength = 80;

% Working out the necessary sampling frequency
inte = floor(length(direct_deaths)/newlength);
    
% A vector of the number of days into conflict for plot
t = [inte:inte:length(direct_deaths)];
if length(t) >= newlength + 1
    newlength = length(t);
    %t = t(1:newlength);
    %t = t(length(t)-newlength+1:end);
end

% Setting out the vectors to be filled by the sampled points
direct_sampled = zeros(newlength,1);
indirect_sampled = zeros(newlength,1);

% Filling the sampled vector
for i = 1:newlength
    direct_sampled(i) = direct_deaths(i*inte);
    indirect_sampled(i) = indirect_deaths(i*inte);
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

%options = optimoptions('fminunc','Algorithm','trust-region','GradObj','on','DerivativeCheck','on','TolX',10);

% Turning on gradient search and reducing the tolerance for the first round
% of 'rough' optiisations
options = optimoptions('fminunc','Algorithm','trust-region','GradObj','on','TolX',1);

% This is to suppress the output of the optimizer. It's irritating.
%options.Display = 'off';

% This is a function that limits its output between -1 and 1, it will be
% used to constrain the correlation parameter.
logit = @(x) (2./(1+exp(-x)))-1;
ilogit = @(x) -log((2./(x+1))-1);          % Inverse of function
dlogit = @(x) (1-x.^2)./(2);               % Derivative of function

% This section defines the starting points for the first 'rough' round of
% optimisation.

num = 30;                                   % Number of start points
precision = 10^-4;                          % Precision of start points
steps = 1/precision;

% Constraining the locations of the selected start points
h_upper = 3;                                % Maximum height start point
ch_upper = 20;
l_lower = 40;                               % Lowest length start point
l_upper = 140;                              % Maximum lengthstart point
n_upper = 0.01;                             % Maximum noise start point

% Compiling vectors of possible start points
h = [h_upper/steps:h_upper/steps:h_upper];  % Height
ch = [ch_upper/steps:ch_upper/steps:ch_upper]; 
l_step = (l_upper-l_lower)/steps;         
l = [l_lower+l_step:l_step:l_upper];        % Length
n = [n_upper/steps:n_upper/steps:n_upper];  % Noise

% Setting up the latin hyper-cube, containing numbers from 0 to the #steps
X = steps*lhsdesign(num,7);

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
    samples(3) = log(ch(X(i,3)));
    samples(4) = l(X(i,4));
    samples(5) = log(n(X(i,5)));
    samples(6) = log(n(X(i,6)));
    %{
    samples(1) = log(1.5);
    samples(2) = log(1.4);
    samples(3) = log(1.3);
    samples(4) = l(60);
    samples(5) = log(0.001);
    samples(6) = log(0.002);
    samples(7) = ilogit(0.8);
    %}

    [xc_chosen,li1] = find_xc([transpose(samples);logy(:,1);logy(:,2)]);
    
    %optionsi = optimoptions('fminunc','Algorithm','trust-region','GradObj','on','TolX',10);
    
    try [x,fval] = fminunc(@product,[transpose(samples);logy(:,1);...
            logy(:,2)],options);
        %{
        [xc_chosen,li2] = find_xc(x);
        
        options = optimoptions('fminunc','Algorithm','trust-region','GradObj','on','TolX',10);

        [x,fval] = fminunc(@product,[transpose(samples);logy(:,1);...
            logy(:,2)],options);        
        %}
    catch
        fprintf('Undefined at initial point')
        fval = 10^8;

    end
       
    fval
    
    if fval < best
        best = fval;                      % Replace best        
        x_ = x;
        li_ = li1;
        %xc_ = xc_chosen;
        
    end

end

%xc_chosen = xc_;

best
%{
x=x_;

tol = [10,1,10^-2,10^-4];

for i = 1:length(tol)

    xc_chosen = find_xc(x)
    options = optimoptions('fminunc','Algorithm','trust-region','GradObj','on','TolX',tol(i));

    [x,fval] = fminunc(@product,[transpose(samples);logy(:,1);...
        logy(:,2)],options);
end
%}

[xc_chosen,li3] = find_xc(x_);
optionsnew = optimoptions('fminunc','Algorithm','trust-region','GradObj','on','TolX',1e-4);
[x,fval] = fminunc(@product,x_,optionsnew);

%[x,fval] = fminunc(@product,x,optionsnew);

rho1_final = exp(x(1));              % Replace hyper-parameters
rho2_final = exp(x(2));
xc_final = t(xc_chosen);
cf_final = exp(x(3));
l_final = x(4);
s_final1 = exp(x(5));
s_final2 = exp(x(6));

v_ = x(7:end);                    % Replace poisson rates

% Split into seperate data streams
v_final = [v_(1:newlength),v_(newlength+1:end)];

fval

var = hessian_diag(v_);

% Calculating the variance terms
variance = [var(1:newlength),var(newlength+1:end)];

% For the testing script
variances = variance;

rho1_final 
rho2_final              
cf_final
xc_final
l_final
s_final1
s_final2

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

figure
p = t(round(newlength/2):newlength-10);
plot(p,li3)
title('Variation of Likelihood with Changepoint Location')
xlabel('Changepoint /Days')
ylabel('Likelihood')
%{
subplot(3,1,1)
plot([1:length(li3)],li_(1,:))
subplot(3,1,2)
plot([1:length(li3)],li_(2,:))
subplot(3,1,3)
plot([1:length(li3)],li3)
%}

% End of this section



%-------------------------------
% END OF MODEL PLOTTING SECTION
%-------------------------------


%-------------------------------------------------------
% START OF LOCAL FUNCTION SECTION
%-------------------------------------------------------

    function [xc_best,li] = find_xc(x)
        rho_1 = exp(x(1));
        rho_2 = exp(x(2));
        cf = exp(x(3));                % Change point factor
        x_l = x(4);                  % Length scale
        x_s1 = exp(x(5));              % Direct noise variance
        x_s2 = exp(x(6));              % Indirect noise variance
        
        v = x(7:end);                  % Predicted poisson rates

        noise = [(x_s1+10^-12)*eye(newlength,newlength),...
            zeros(newlength,newlength);zeros(newlength,newlength),...
            (x_s2+10^-12)*eye(newlength,newlength)];

        % difference vector of the observed values
        d = v-[mean(1)*ones(newlength,1);mean(2)*ones(newlength,1)];
                            
        K = cov_matrix2(t,t,x_l);
        
        Kf = rho_1*K;
        
        Kg = rho_2*K;

        fbest = 10^8;
        
        %range = round(newlength/3):newlength-3;
        range = round(newlength/2):newlength-10;
        
        li = zeros(1,length(range));
        ri = 1;
        
        for xc = range
            
            Kg2 = [Kg(1:xc,1:xc),sqrt(cf)*Kg(1:xc,xc+1:end);...
                sqrt(cf)*Kg(xc+1:end,1:xc),cf*Kg(xc+1:end,xc+1:end)];
        
            cov = [Kf+Kg2,Kf;Kf,Kf]+noise;

            % Cholesky decomposition to avoid inverting covariance
            L = chol(cov,'lower');
            R = transpose(L);
 
            % The quantity we are trying to minimize - the negatie of the log
            % likelihood containing only the terms which change with v
            ftrial = sum(exp(v))-transpose(v)*[y(:,1);y(:,2)]+0.5*log(det(cov))...
            +0.5*transpose(d)*(R\(L\d));
        
            if ftrial <= fbest
                fbest = ftrial;
                xc_best = xc;
            end
            
            li(ri) = -ftrial;
            ri = ri+1;
           
        end
        
        li = li/sum(li);
 
    end


% Function which calculates selected terms of the posterior numerator.
    function [f,g] = product(x)
        rho_1 = exp(x(1));
        rho_2 = exp(x(2));
        cf = exp(x(3));                % Change point factor
        x_l = x(4);                  % Length scale
        x_s1 = exp(x(5));              % Direct noise variance
        x_s2 = exp(x(6));              % Indirect noise variance
        
        xc = xc_chosen;
        
        v = x(7:end);                  % Predicted poisson rates

        noise = [(x_s1+10^-12)*eye(newlength,newlength),...
            zeros(newlength,newlength);zeros(newlength,newlength),...
            (x_s2+10^-12)*eye(newlength,newlength)];

        % difference vector of the observed values
        d = v-[mean(1)*ones(newlength,1);mean(2)*ones(newlength,1)];
        
        K = cov_matrix2(t,t,x_l);
        
        Kf = rho_1*K;
        
        Kg = rho_2*K;
        
        Kg = [Kg(1:xc,1:xc),sqrt(cf)*Kg(1:xc,xc+1:end);...
            sqrt(cf)*Kg(xc+1:end,1:xc),cf*Kg(xc+1:end,xc+1:end)];
        
        cov = [Kf+Kg,Kf;Kf,Kf]+noise;
        
        % Cholesky decomposition to avoid inverting covariance
        L = chol(cov,'lower');
        R = transpose(L);
        
        f = sum(exp(v))-transpose(v)*[y(:,1);y(:,2)]+0.5*log(det(cov))...
            +0.5*transpose(d)*(R\(L\d));
        
        % Gradients for the poisson rate values
        g = R\(L\d)+exp(v)-[y(:,1);y(:,2)];
        
        % DERIVATIVES FOR THE HYPERPARAMETERS
        
        covdr1 = [K,K;K,K];
        
        covdr2 = [Kg/rho_2,zeros(newlength,newlength);...
            zeros(newlength,newlength),zeros(newlength,newlength)];
        
        % Now compute the actual gradients for all of the output scales
        
        gr1 = 0.5*rho_1*(-transpose(d)*(R\(L\covdr1))*(R\(L\d))+trace(R\(L\covdr1)));
        gr2 = 0.5*rho_2*(-transpose(d)*(R\(L\covdr2))*(R\(L\d))+trace(R\(L\covdr2)));
               
        dcov = zeros(newlength,newlength);
        
        % Next the gradient for the length scale
        for ci = 1:newlength
            for cj = 1:newlength
                dcov(ci,cj) = (t(ci)-t(cj))^2*K(ci,cj);
            end
        end

        dcov = dcov/(x_l^3);
        
        dKgdl = rho_2*[dcov(1:xc,1:xc),sqrt(cf)*dcov(1:xc,xc+1:end);...
            sqrt(cf)*dcov(xc+1:end,1:xc),cf*dcov(xc+1:end,xc+1:end)];
        
        dKfdl = rho_1*dcov;
        
        dcovl = [dKfdl+dKgdl,dKfdl;dKfdl,dKfdl];
        
        gl = 0.5*(-transpose(d)*(R\(L\dcovl))*(R\(L\d))+trace(R\(L\dcovl)));
        
        % Now the gradients for the noise values
        dcovs1 = [eye(newlength),zeros(newlength,newlength);zeros(newlength,2*newlength)];
        dcovs2 = [zeros(newlength,2*newlength);zeros(newlength,newlength),eye(newlength)];
        
        gs1 = 0.5*x_s1*(-transpose(d)*(R\(L\dcovs1))*(R\(L\d))+trace(R\(L\dcovs1)));
        gs2 = 0.5*x_s2*(-transpose(d)*(R\(L\dcovs2))*(R\(L\d))+trace(R\(L\dcovs2)));
        
        % Now for the change point factor
        
        dKgdcf = [zeros(xc,xc),0.5*cf^(-0.5)*K(1:xc,xc+1:end);...
            0.5*cf^(-0.5)*K(xc+1:end,1:xc),K(xc+1:end,xc+1:end)];
        
        dcovcf = [rho_2*dKgdcf,zeros(newlength,newlength);...
            zeros(newlength,2*newlength)];
        
        gcf = 0.5*cf*(-transpose(d)*(R\(L\dcovcf))*(R\(L\d))+trace(R\(L\dcovcf)));
        
        % Compiling the gradients into one vector
        g = [gr1;gr2;gcf;gl;gs1;gs2;g];
    end

% Function which calculates the diagonal terms of the hessian matrix given
% a set of intensities, and using global hper-parameters. This vecotor will
% be the predicted variances for each timestep. 
    function va = hessian_diag(v)
        
        noise = [(s_final1+10^-12)*eye(newlength,newlength),...
            zeros(newlength,newlength);zeros(newlength,newlength),...
            (s_final2+10^-12)*eye(newlength,newlength)];

        xc = xc_chosen;
        
        K = cov_matrix2(t,t,l_final);
        
        Kf = rho1_final*K;
        
        Kg = rho2_final*K;
        
        Kg = [Kg(1:xc,1:xc),sqrt(cf_final)*Kg(1:xc,xc+1:end);...
            sqrt(cf_final)*Kg(xc+1:end,1:xc),cf_final*Kg(xc+1:end,xc+1:end)];
        
        cov_ = [Kf+Kg,Kf;Kf,Kf]+noise;

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
