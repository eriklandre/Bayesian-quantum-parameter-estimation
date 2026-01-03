function [T,score] = phaseestimation_kcopy(No_initial, No_final, method, ncopies, strategy)
% No_initial: initial number of estimators
% No_final: final number of estimators
% ncopies: number of copies of the channel
% set method = 1, 2, or 3, to choose desired method
% set strategy = 1, 2, or 3, to choose desired strategy (PARALLEL, SEQUENTIAL, GENERAL)

d = 3;

Sz = eye(d);

for i=1:d
    Sz(i,i) = (i-1);
end

score = zeros(No_final-No_initial+1,1);
T     = cell(No_final-No_initial+1,1);

for No=No_initial:No_final
    
    No = No
    
    %%%%%%%%% M1 %%%%%%%%%
    if method==1
        Nh = No;  
        
    %%%%%%%%% M2 and M3 %%%%%%%%%   
    else
        Nh = 5000; 
    end
    
    estimators = 0:2*pi/(No):2*pi;      % uniform distribution of the estimator 
    discretization = 0:2*pi/(Nh):2*pi;  % uniform discretization of the true parameter 
    
    theta_i = estimators(1:No);         
    theta_k = discretization(1:Nh);   

    %p(1:Nh,1) = 1/Nh;       % uniform prior
    p = prior_p0_density(theta_k, 0, 2*pi, 100); % custom prior
    
    %p = normpdf(theta_k,pi,1);  % gaussian prior
    %p = p./sum(p);              % normalization condition
    
    Ck = zeros(d^2,d^2,Nh); 
    for k=1:Nh
        Ck(:,:,k) = kraus2choi(expm(-1i*theta_k(k)*Sz));    % Choi matrices of the true channels
    end

    Xi = zeros(d^(2*ncopies),d^(2*ncopies),No);
    r = zeros(No,Nh);
    for i=1:No
        for k=1:Nh
            r(i,k) = (theta_k(k)-theta_i(i))^2;      % reward function
            %r(i,k) = cos((theta_k(k)-theta_i(i))/2)^2;          % cosine reward function
            Xi(:,:,i) = Xi(:,:,i) + p(k)*r(i,k)*Tensor(Ck(:,:,k),ncopies);      % computing the Xi's
        end
    end

    [T{No-No_initial+1,1},score(No-No_initial+1,1),~] = testeroptimization_sdp_kcopy_seesaw(Xi,[d d],ncopies,strategy,-1);      % SDP optimization to get the optimal tester and score
    
    %%%%%%%%% M3: SEESAW algorithm (takes as input the optimal tester and optimizes over estimators) %%%%%%%%% 
    if method==3    
        
        T_temp = T{No-No_initial+1,1};
        
        gap = 1;
        precision = 10^(-6);
        rounds = 0;
        old_score = score(No-No_initial+1,1);
        flag_value = 0;
        
        while gap>precision
            
            rounds = rounds + 1;
            
            [estimators] = estimator_optimization(p,T_temp,Ck,theta_k,ncopies);      % optimization over the estimators
            
            theta_i = estimators;
            
            Xi = zeros(d^(2*ncopies),d^(2*ncopies),No);
            r  = zeros(No,Nh);
            for i=1:No
                for k=1:Nh
                    r(i,k) = cos((theta_k(k)-theta_i(i))/2)^2;      % reward function
                    Xi(:,:,i) = Xi(:,:,i) + p(k)*r(i,k)*Tensor(Ck(:,:,k),ncopies);      % computing the Xi's
                end
            end
            
            [T_temp,score_temp,flag] = testeroptimization_sdp_kcopy_seesaw(Xi,[d d],ncopies,strategy,1);        % SDP optimization to get the optimal tester and score
            if flag.problem~=0
                sdp_problem = flag.problem
                info        = flag.info
                flag_value = 1;
                %pause
            end
            
            gap       = abs(old_score - score_temp);
            old_score = score_temp;
            %pause
        end
        
       T{No-No_initial+1,1} = T_temp;
       score(No-No_initial+1,1) = score_temp;            
    end
       
end

end

function [estimators] = estimator_optimization(p,T,Ck,theta_k,ncopies)

Nh = max(size(p));
No = size(T,3);

post = zeros(Nh,No);
avg_sin = zeros(No,1);
avg_cos = zeros(No,1);
estimators = zeros(No,1);

for k = 1:Nh
    for i = 1:No
        post(k,i) = p(k)*real(trace(T(:,:,i)*Tensor(Ck(:,:,k), ncopies)));
    end
end
post = post./sum(post,1);

for i = 1:No
    for k = 1:Nh
        avg_sin(i,1) = avg_sin(i,1) + post(k,i)*sin(theta_k(k));
        avg_cos(i,1) = avg_cos(i,1) + post(k,i)*cos(theta_k(k));
    end
    if avg_cos(i,1)>=0 
       estimators(i,1) = atan(avg_sin(i,1)/avg_cos(i,1));
    else 
       estimators(i,1) = atan(avg_sin(i,1)/avg_cos(i,1)) + pi;
    end
    if estimators(i,1)<0
        estimators(i,1) = estimators(i,1) + 2*pi;
    end
end

end 

function C = kraus2choi(K)

d_out  = size(K,1);
d_in   = size(K,2);
nkraus = size(K,3);

psi = zeros(d_in^2,1);
vec = eye(d_in);

for i=1:d_in
    psi = psi + kron(vec(:,i),vec(:,i));
end

C = zeros(d_in*d_out,d_in*d_out);

for i=1:nkraus
    C = C + kron(eye(d_in),K(:,:,i))*psi*(psi')*kron(eye(d_in),K(:,:,i)');
end

end

function p = prior_p0_density(h, hmin, hmax, alpha)     % gives the prior distribution in Eq. ()
    % h: vector of points where the prior is evaluated
    % hmin, hmax: lower and upper bound of the interval
    % alpha: shape parameter (larger alpha -> more peaked prior)

    norm_const = (hmax - hmin) * (exp(alpha/2) * besseli(0, alpha/2) - 1);      % normalization constantv

    arg = pi * (h - hmin) / (hmax - hmin);
    num = exp(alpha * (sin(arg).^2)) - 1;

    p = num ./ norm_const;  % unnormalized prior
    p = p ./ sum(p);        % normalization condition
    
end