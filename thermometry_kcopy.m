function [T,score] = thermometry_kcopy(t,No_initial,No_final,ncopies,strategy)
% ncopies: number of copies of the channel
% set strategy = 1, 2, or 3, to choose desired strategy (PARALLEL, SEQUENTIAL, GENERAL)
% t: fixed time for the unitary evolution

d = 2;

Tmin = 0.1;     % minimum temperature
Tmax = 2;       % maximum temperature

Nt = 100;       % number of time steps

time = 0:(1/(Nt-1)):1;      % time vector

score = zeros(No_final-No_initial+1,Nt);
T     = cell(No_final-No_initial+1,Nt);

for No=No_initial:No_final
   
    No = No
    Nh = 1000; 
   
    theta_i(1:No) = Tmin:(Tmax-Tmin)/(No-1):Tmax;       % uniform distribution of the estimator
    theta_k(1:Nh) = Tmin:(Tmax-Tmin)/(Nh-1):Tmax;       % uniform discretization of the true parameter
    
    %p(1:Nh,1) = 1/Nh;      % uniform prior
    p = prior_p0_density(theta_k, Tmin, Tmax, -100); % custom prior
    
    % gaussian prior %
    % p = normpdf(theta_k,0,1);
    % p = p./sum(p);
     
    r = zeros(No,Nh);
    for i=1:No
        for k=1:Nh
            %r(i,k) = cos((theta_k(k)-theta_i(i))/2)^2; 
            r(i,k) = ((theta_k(k)-theta_i(i))/theta_k(k))^2;    % cost function
        end
    end
        
    Ck = zeros(d^2,d^2,Nh);
    for k=1:Nh
        Ck(:,:,k) = ChoiOperatorThermo(theta_k(k),0.1,2,time(t));       % Choi matrices of the true channels
    end
    
    Xi = zeros(d^(2*ncopies),d^(2*ncopies),No);
    for i=1:No
        for k=1:Nh
            Xi(:,:,i) = Xi(:,:,i) + p(k)*r(i,k)*Tensor(Ck(:,:,k),ncopies);      % computing the Xi's
        end
    end
    
    [T{No-No_initial+1,t},score(No-No_initial+1,t),~] = testeroptimization_sdp_kcopy_seesaw(Xi,[d d],ncopies,strategy,-1);     % Tester optimization
    
    % Seesaw algorithm
     T_temp = T{No-No_initial+1,t};
     
     gap = 1;
     precision = 10^(-6);
     rounds(t,1) = 0;
     old_score = score(No-No_initial+1,t);
     flag_value = 0;
     
     while gap>precision
         
         rounds(t,1) = rounds(t,1) + 1;
         
         % Step 1: optimizing the estimators
         
         [estimators] = estimator_optimization(p,T_temp,Ck,theta_k,ncopies);     % Estimator optimization
         
         theta_i = estimators;
         Xi = zeros(d^(2*ncopies),d^(2*ncopies),No);
         r = zeros(No,Nh);
         for i=1:No
             for k=1:Nh
                 %r(i,k) = cos((theta_k(k)-theta_i(i))/2)^2; % cosine reward (like in phase estimation)
                 r(i,k) = ((theta_k(k)-theta_i(i))/theta_k(k))^2; % relative error
                 %r(i,k) = ((theta_k(k)-theta_i(i)))^2; % MSE
                 Xi(:,:,i) = Xi(:,:,i) + p(k)*r(i,k)*Tensor(Ck(:,:,k),ncopies);
             end
         end
         
         % Step 2: reoptimizing the tester
         
         [T_temp,score_temp,flag] = testeroptimization_sdp_kcopy_seesaw(Xi,[d d],ncopies,strategy,-1);      % Reoptimizing the tester
         
         if flag.problem~=0
             sdp_problem = flag.problem
             info        = flag.info
             flag_value = 1;
             %pause
         end
         
         gap         = abs(old_score - score_temp);
         old_score   = score_temp;
         %pause
     end
     
     T{No-No_initial+1,t} = T_temp;
     score(No-No_initial+1,t) = score_temp;
end

end

function [estimators] = estimator_optimization(p,T,Ck,theta_k,ncopies)
% Estimator optimization given relative MSE as cost function
Nh = max(size(p));
No = size(T,3);

post = zeros(Nh,No);
estimators = zeros(No,1);

for k = 1:Nh
    for i = 1:No
        post(k,i) = p(k)*real(trace(T(:,:,i)*Tensor(Ck(:,:,k),ncopies)));
    end
end
post = post./sum(post,1);

for i = 1:No
    num_1 = 0;
    den_1 = 0;
    for k = 1:Nh
        num_1 = num_1 + post(k,i) * (1/theta_k(k));
        den_1 = den_1 + post(k,i) * (1/(theta_k(k)^2));
    end
    estimators(i,1) = num_1 / den_1;
end

end

function J = ChoiOperatorThermo(T,eps,g,t)

% explicit form of Choi state
N = 1/(exp(eps/T) - 1);
gamma = g*(2*N+1);
a = (N+N*exp(-gamma*t)+1)/(2*(2*N+1));
b = ((1-exp(-gamma*t))*(N+1))/(2*(2*N+1));
c = (N-N*exp(-gamma*t))/(2*(2*N+1));
d = (exp(-gamma*t)*(N+1)+N)/(2*(2*N+1));

diag_terms = [a b c d];
off_diag_term = exp(-(gamma*t)/2)/2;

J = diag(diag_terms);
J(1,4) = off_diag_term;
J(4,1) = off_diag_term';

% change choi states to choi matrices + swap parties for consistency with the notes
J = 2 * J; %this is to include a 2 factor missing in this definition of the Choi
J = Swap(J); %this is to use the definition C_theta [|i><j|] = \sum_ij |i><j| x E[|i><j|] and not the other way around

end

function [estimators] = estimator_optimization_cos(p,T,Ck,theta_k,ncopies) % use this function if the reward is cos
%%%% estimator optimization for phase estimation %%%%

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
