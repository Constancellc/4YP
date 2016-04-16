function generate_test
% for known hyper-parameters, working out da total posterior. like a boss

a = [0.9901, 0.1464, 3.9213, 44.6529, 40.5193, 0.0005, 418.0000];

% GETTING RAW DATA
direct_deaths = csvread('Data/Direct Frequencies.csv');
indirect_deaths = csvread('Data/Indirect Frequencies.csv');
ndays = length(direct_deaths);
days = transpose(1:ndays);


log_full = zeros(2*ndays,1);

for i = 1:ndays
    if direct_deaths(i) == 0
        log_full(i) = 0;
    else
        log_full(i) = log(direct_deaths(i));
    end
    
    if indirect_deaths(i) == 0
        log_full(i+ndays) = 0;
    else
        log_full(i+ndays) = log(direct_deaths(i));
    end
end
%}

% SAMPLING DATA
newlength = 80;

% Working out the necessary sampling frequency
inte = floor(length(direct_deaths)/newlength);
    
% A vector of the number of days into conflict for plot
t = [inte:inte:length(direct_deaths)];
if length(t) >= newlength + 1
    newlength = length(t);
end

% Setting out the vectors to be filled by the sampled points
y = zeros(newlength,2);

% Filling the sampled vector
for i = 1:newlength
    y(i,1) = direct_deaths(i*inte);
    y(i,2) = indirect_deaths(i*inte);
end

Y = [y(:,1);y(:,2)];

% PICKING OUR BRAND NEW TEST TIMES

test = [55,101,631,812,1210];
testlength = length(test);

t_ = [t,test];

% Do i actually want to do this?
t_ = sort(t_);

test_index = [];

for i = 1:length(test)
    goal = test(i);
    
    for k = 1:length(t_)
        if t_(k) == goal
            test_index = [test_index;k];
        end
    end
end

test_index



% FINDING CHOSEN HYPERS
rho1 = a(1);
rho2 = a(2);
cf = a(3);
l1 = a(4);
l2 = a(5);
s = a(6);
xc = a(7);
index = xc/inte;

best = 100;

for i = 1:length(t_)
    gap = abs(xc-t_(i));
    
    if gap < best
        best = gap;
        test_xc = i;
    end
end

% MEAN VECTOR FOR ALL OF TIME
mew = [mean(1)*ones(newlength+testlength,1);...
    mean(2)*ones(newlength+testlength,1)];

% FINDING FUCKING MASSIVE COVARIANCE
noise = (s+10^-6)*eye(2*(newlength+testlength),2*(newlength+testlength));
K = cov_matrix2(t_,t_,l1);
K2 = cov_matrix2(t_,t_,l2);
Kf = rho1*K;
Kg = rho2*K2;
Kg = [Kg(1:test_xc,1:test_xc),sqrt(cf)*Kg(1:test_xc,test_xc+1:end);...
    sqrt(cf)*Kg(test_xc+1:end,1:test_xc),cf*Kg(test_xc+1:end,test_xc+1:end)];
cov = [Kf+Kg,Kf;Kf,Kf]+noise;



L = chol(cov,'lower');
R = transpose(L);

% Sorting out the base for the likelihood

likelihood = 0;

for i=1:newlength*2
    likelihood = likelihood -log(factorial(Y(i)));
end

% THE MOTHERFUCKING MONTECARLO SAMPLING FOR THE MARGINAL

% FIRST GOTTA GET DAT MEAN VECTOR
mewo = [mean(1)*ones(newlength,1);mean(2)*ones(newlength,1)];

% FINDING LESS MASSIVE COVARIANCE
noiseo = (s+10^-12)*eye(2*newlength,2*newlength);
Ko = cov_matrix2(t,t,l1);
K2o = cov_matrix2(t,t,l2);
Kfo = rho1*Ko;
Kgo = rho2*K2o;
Kgo = [Kgo(1:index,1:index),sqrt(cf)*Kgo(1:index,index+1:end);...
    sqrt(cf)*Kgo(index+1:end,1:index),cf*Kgo(index+1:end,index+1:end)];
covo = [Kfo+Kgo,Kfo;Kfo,Kfo]+noiseo;

Lo = chol(covo,'lower');
Ro = transpose(Lo);

N = 1000;
int_approx = 0;

for i = 1:N
    v_t = mvnrnd(mewo,covo);
    int_approx = int_approx+like(v_t);
end

marg = int_approx/N

    function l = like(v_t)
        % 4 DA MONTECARLO
        l = likelihood-sum(exp(v_t))+v_t*Y;
    end

% AND FOR THE GRAND FINALE, THE REASON WE'RE ALL HERE LADIES + GENTS

% SEARCHING FOR THE MAXIMUM LIKELY VALUES 
options = optimoptions('fminunc','Algorithm','trust-region','GradObj','on');

start = zeros(2*length(t_),1);

for i = 1:length(t_)
    start(i) = log_full(t_(i));
    start(i+length(t_)) = log_full(t_(i)+ndays);
end

posterior(start)

[v_final,fval] = fminunc(@posterior,start,options)

var = hessian_diag(v_final);

% HACK HACK HACK

v_final2 = [v_final(1:length(t_)),v_final(length(t_)+1:end)];

% Calculating the variance terms
variance = [var(1:length(t_)),var(length(t_)+1:end)];

% Working out vectors of the mean +/- 2 std 
mean_plus = zeros(2,length(t_));
mean_minus = zeros(2,length(t_));

for k = 1:length(t_)
    for h = 1:2
        mean_plus(h,k) = exp(v_final2(k,h) + 2*sqrt(variance(k,h)));
        mean_minus(h,k) = exp(v_final2(k,h) - 2*sqrt(variance(k,h)));
    end
end

X = [t_,fliplr(t_)];
Y1 = [mean_minus(1,:),fliplr(mean_plus(1,:))];
Y2 = [mean_minus(2,:),fliplr(mean_plus(2,:))];

% HACK HACK HACK

counts = exp(v_final);

testing = [];
Lo = [];
Up = [];
xt = [];

for i = 1:length(test_index)
    
    dv = v_final(test_index(i));
    iv = v_final(test_index(i)+length(t_));
    
    dcount = counts(test_index(i));
    icount = counts(test_index(i)+length(t_));
    
    testing = [testing,[dcount;icount]];
    
    vard = var(test_index(i));
    vari = var(test_index(i)+length(t_));
    
    dup = exp(dv+2*sqrt(vard));
    iup = exp(iv+2*sqrt(vari));
    
    Up = [Up,[dup;iup]];
    
    dlo = exp(dv-2*sqrt(vard));
    ilo = exp(iv-1*sqrt(vari));
    
    Lo = [Lo,[dlo;ilo]];
    
    xt = [xt,t_(test_index(i))];
end

testing = [testing(1,:),testing(2,:)];
Lo = [Lo(1,:),Lo(2,:)];
Up = [Up(1,:),Up(2,:)];
Up = Up-testing;
Lo = testing-Lo;

shade = fill(X,Y1,'b');               % Fills the confidence region   
set(shade,'facealpha',.1)             % Sets the region transparent
set(shade,'EdgeColor','None')
hold on
shade2 = fill(X,Y2,'g');              
set(shade2,'facealpha',.1)           
set(shade2,'EdgeColor','None')
plot(t_,[counts(1:length(t_)),counts(length(t_)+1:end)])
hold on
plot(t,y,'x')
hold on
errorbar([xt,xt],testing,Lo,Up,'ro')
%plot(xt,testing,'r*')

    function [f,g] = posterior(v)
        % v is going to be the full set of intensity values

        % v should be 2*newlength+testlength long

        % V will correspond to the observed sample times
        
        j2 = 1;

        for j = 1:(newlength+testlength)
            if any(test_index == j)
                % Do nothing
            else
                V(j2) = v(j);
                V(j2+newlength) = v(j+newlength);
                j2 = j2+1;
            end
        end

        eV = exp(V);
        
        gi = transpose(Y) - eV;

        % we can now work out da massive difference function
        d = v - mew;

        p_D_v = likelihood + V*Y-sum(eV);

        p_v_theta = -length(t_)*log(2*pi) - 0.5*log(det(cov)) - 0.5*transpose(d)*(R\(L\d));

        f = p_D_v + p_v_theta - marg;
        
        g = -(R\(L\d));
        
        j3 = 1;
        
        for j = 1:(newlength+testlength)
            if any(test_index == j)
                % Do nothing
            else
                g(j) = g(j) + gi(j3);
                g(j+newlength+testlength) = g(j+newlength+testlength) + gi(j3+newlength);
                j3 = j3+1;
            end
        end
        
        % Because min not max
        f = -f;
        g = - g;
        
    end

    function va = hessian_diag(v)

        icov = R\(L\eye(2*length(t_)));
        
        A = eye(2*length(t_));
        
        for ai = 1:length(t_)*2
            for aj = 1:length(t_)*2
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

