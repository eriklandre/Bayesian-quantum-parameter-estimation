function score_adaptive = thermometry_greedy(No, k_copies, n_monte_carlo, t)
% No: number of outcomes of the measurement
% k_copies: number of uses of the channel to use in the Greedy algorithm
% n_monte_carlo: number of Monte Carlo samples
% t: time parameter of the thermometry channel

d = 2;

Tmin = 0.1;     % minimum temperature
Tmax = 2;       % maximum temperature
Nh = 2500;      % number of hypotheses
Nt = 100;       % number of time steps

time = 0:(1/(Nt-1)):1;  % time vector

theta_k(1:Nh) = Tmin:(Tmax-Tmin)/(Nh-1):Tmax;       % true parameter discretization

p_initial(1:Nh,1) = 1/Nh;      % uniform prior

Ck = zeros(d^2,d^2,Nh);
for k=1:Nh
    Ck(:,:,k) = ChoiOperatorThermo(theta_k(k),0.1,2,time(t));       % Choi of the true parameter theta_k
end

score_adaptive = run_adaptive_monte_carlo(p_initial, theta_k, Ck, No, k_copies, n_monte_carlo, t);      % Run the MC simulation
fprintf('Adaptive score: %.6f\n', score_adaptive); % score

end

function score_adaptive = run_adaptive_monte_carlo(p_updated, theta_k, Ck, No, k_copies, n_monte_carlo, t)

d = 2;
Tmin = 0.1;    
Tmax = 2;       

Nt = 100;      

time = 0:(1/(Nt-1)):1;

Nh = length(p_updated);

final_scores = zeros(n_monte_carlo, 1);

for mc = 1:n_monte_carlo
    fprintf('Monte Carlo run %d of %d...\n', mc, n_monte_carlo);
    theta_true = sample_true_parameter(p_updated, theta_k);      % random sample of the true parameter

    p_current = p_updated;
    theta_i_prev = [];
    for copy = 1:k_copies
        if copy == 1
            theta_i(1:No) = Tmin:(Tmax-Tmin)/(No-1):Tmax;
        else
            fprintf('[MC %d] Copy 2 START - estimators:\n', mc);
            theta_i = theta_i_prev;
        end
        
        Xi = zeros(d^2,d^2,No);
        r = zeros(No,Nh);
        for i=1:No
            for k=1:Nh
                %r(i,k) = ((theta_k(k)-theta_i(i)))^2;    % cost function (MSE)
                r(i,k) = ((theta_k(k)-theta_i(i))/theta_k(k))^2; % cost function (relative error)
                Xi(:,:,i) = Xi(:,:,i) + p_current(k)*r(i,k)*Ck(:,:,k);      % computing the Xi's
            end
        end
        
        [T_adaptive, current_score, ~] = testeroptimization_sdp_kcopy_seesaw(Xi,[d d],1,2,-1);      % SDP optimization to get the optimal tester and score

        % SEESAW algorithm (takes as input the optimal tester and optimizes over estimators)

        gap = 1;
        precision = 10^(-4); % decrease to 10^(-6) if needed
        rounds = 0;
        old_score = current_score;
        flag_value = 0;

        while gap>precision
            
            rounds = rounds + 1;
            
            [estimators] = estimator_optimization(p_current,T_adaptive,Ck,theta_k);         % optimization over the estimators
            
            theta_i = estimators;
            Xi = zeros(d^2,d^2,No);
            r = zeros(No,Nh);
            for i=1:No
                for k=1:Nh
                    %r(i,k) = ((theta_k(k)-theta_i(i)))^2;  % MSE
                    r(i,k) = ((theta_k(k)-theta_i(i))/theta_k(k))^2; % relative error
                    Xi(:,:,i) = Xi(:,:,i) + p_current(k)*r(i,k)*Ck(:,:,k);
                end
            end
            
            [T_temp,score_temp,flag] = testeroptimization_sdp_kcopy_seesaw(Xi,[d d],1,2,-1);        % SDP optimization to get the optimal tester and score
            
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

        T_adaptive   = T_temp;
        current_score = score_temp;
        theta_i_prev = theta_i;
        
        C_true = ChoiOperatorThermo(theta_true,0.1,2,time(t));      % Choi operator of the true parameter (randomly sampled at the beginning)
        probs = zeros(No, 1);
        for i = 1:No
            probs(i) = real(trace(T_adaptive(:,:,i) * C_true));     % Simulate measurement with true parameter
        end
        probs = probs / sum(probs);
        
        % Sample random measurement outcome
        cumulative_probs = cumsum(probs);
        rand_val = rand();
        outcome_idx = find(cumulative_probs >= rand_val, 1, 'first');

        if copy < k_copies
            % Update posterior (Bayes rule)
            likelihood = zeros(Nh,1);
            for k = 1:Nh
                likelihood(k) = real(trace(T_adaptive(:,:,outcome_idx) * Ck(:,:,k)));
            end
            p_current = likelihood .* p_current;
            p_current = p_current / sum(p_current);

        else
            % LAST COPY: realized cost for this run
            theta_hat = theta_i(outcome_idx);          % estimator associated to the observed outcome
            final_scores(mc) = ((theta_true - theta_hat)/theta_true)^2;   % relative error cost
        end
    end
end

score_adaptive = mean(final_scores); % average over all MC samples, as we are randomly sampling the true parameter

end

function theta_true = sample_true_parameter(p, theta_k)
    % p: prior distribution
    % theta_k: grid of hypotheses

    p = p(:);       
    c = cumsum(p / sum(p));  
    u = rand();
    idx = find(c >= u, 1, 'first');
    theta_true = theta_k(idx);
end

function [estimators] = estimator_optimization(p,T,Ck,theta_k)

Nh = max(size(p));
No = size(T,3);

post = zeros(Nh,No);
estimators = zeros(No,1);

for k = 1:Nh
    for i = 1:No
        post(k,i) = p(k)*real(trace(T(:,:,i)*Ck(:,:,k)));
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

function [estimators] = estimator_optimization_MSE(p,T,Ck,theta_k) % use this function if the reward is MSE

Nh = max(size(p));
No = size(T,3);

post = zeros(Nh,No);
estimators = zeros(No,1);

for k = 1:Nh
    for i = 1:No
        post(k,i) = p(k)*real(trace(T(:,:,i)*Ck(:,:,k)));
    end
end
post = post./sum(post,1);

for i = 1:No
    for k = 1:Nh
        estimators(i,1) = estimators(i,1) + post(k,i)*theta_k(k);
    end
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
