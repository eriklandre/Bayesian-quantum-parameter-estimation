function [T,score] = phaseestimation_and_noise_kcopy(No_initial, No_final, ncopies, strategy, p)
% strategy = 1,2,3 for PAR, SEQ, GEN
% p: noise parameter between 0 and 1 
d = 2;

score = zeros(No_final-No_initial+1,1);
T     = cell(No_final-No_initial+1,1);

for No=No_initial:No_final
    
    No = No
    Nh = 10000; 

    estimators = 0:2*pi/(No):2*pi;      % uniform distribution of the estimator 
    discretization = 0:2*pi/(Nh):2*pi;  % uniform discretization of theta 

    theta_i = estimators(1:No);         % will use only first No values, not including 2pi
    theta_k = discretization(1:Nh);     % will use only first Nh values, not including 2pi

    %p_initial(1:Nh,1) = 1/Nh;       % uniform prior
    p_initial = prior_p0_density(theta_k, 0, 2*pi, -100); % custom prior
    p_initial = p_initial(:) ./ sum(p_initial); % normalization condition

    %p = normpdf(theta_k,pi,1);  % gaussian prior
    %p = p./sum(p); % normalization condition
    
    Ck = zeros(d^2,d^2,Nh); 
    for k=1:Nh
        U  = diag([exp(-1i*theta_k(k)/2), exp(1i*theta_k(k)/2)]); % Kraus operator of the unitary channel
        % Kraus operators of the AD channel
        A0 = diag([1, sqrt(1-p)]);
        A1 = [0, sqrt(p); 0, 0];
        % Total Kraus operators
        KrausT{1,1} = A0*U; 
        KrausT{2,1} = A1*U;
        Ck(:,:,k) = ChoiMatrix(KrausT); % Choi operators of the true parameters
    end

    Xi = zeros(d^(2*ncopies),d^(2*ncopies),No);
    r = zeros(No,Nh);
    for i=1:No
        for k=1:Nh
            r(i,k) = cos((theta_k(k)-theta_i(i))/2)^2;          % cos reward function
            Xi(:,:,i) = Xi(:,:,i) + p_initial(k)*r(i,k)*Tensor(Ck(:,:,k),ncopies);
        end
    end

    [T{No-No_initial+1,1},score(No-No_initial+1,1),~] = testeroptimization_sdp_kcopy_seesaw(Xi,[d d],ncopies,strategy,1); % optimizing the tester
    
    % Seesaw algorithm to optimize tester and estimators
    
    T_temp = T{No-No_initial+1,1};
    
    gap = 1;
    precision = 10^(-6);
    rounds = 0;
    old_score = score(No-No_initial+1,1);
    flag_value = 0;
    
    while gap>precision
        
        rounds = rounds + 1;
                    
        % Step 1: optimizing the estimator
        
        [estimators] = estimator_optimization(p_initial,T_temp,Ck,theta_k,ncopies);
        
        theta_i = estimators;
        
        Xi = zeros(d^(2*ncopies),d^(2*ncopies),No);
        r  = zeros(No,Nh);
        for i=1:No
            for k=1:Nh
                r(i,k) = cos((theta_k(k)-theta_i(i))/2)^2;
                Xi(:,:,i) = Xi(:,:,i) + p_initial(k)*r(i,k)*Tensor(Ck(:,:,k),ncopies);
            end
        end
        
        %Step 2: reoptimizing the tester
        
        [T_temp,score_temp,flag] = testeroptimization_sdp_kcopy_seesaw(Xi,[d d],ncopies,strategy,1);
        if flag.problem~=0
            sdp_problem = flag.problem
            info        = flag.info
            flag_value = 1;
            %pause
        end
        
        gap       = abs(old_score - score_temp); % as long as the difference between the scores of different rounds is not smaller than our precision, we keep running the seesaw
        old_score = score_temp;
        %pause
    end
    
    T{No-No_initial+1,1} = T_temp;
    score(No-No_initial+1,1) = score_temp;            
       
end

end

function [estimators] = estimator_optimization(p,T,Ck,theta_k,ncopies)
% Estimator optimization for phase estimation

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

function p = prior_p0_density(h, hmin, hmax, alpha)
% Custom probability distribution
    norm_const = (hmax - hmin) * (exp(alpha/2) * besseli(0, alpha/2) - 1);
    arg = pi * (h - hmin) / (hmax - hmin);
    num = exp(alpha * (sin(arg).^2)) - 1;

    p = num ./ norm_const;
    p = p ./ sum(p);
end
