function score_adaptive = phaseestimation_greedy(No, k_copies, n_monte_carlo)
% No: number of outcomes of the measurement
% k_copies: number of uses of the channel to use in the Greedy algorithm
% n_monte_carlo: number of Monte Carlo samples

d = 3;

Sz = eye(d);

for i=1:d
    Sz(i,i) = (i-1);        % constructing the spin collective operator
end

Nh = 1000;      % number of hypotheses

discretization = 0:2*pi/(Nh):2*pi;
theta_k = discretization(1:Nh);     % true parameter distribution

%p_initial(1:Nh,1) = 1/Nh;      % uniform prior
p_initial = prior_p0(theta_k, 0, 2*pi, 100);    % custom prior

Ck = zeros(d^2,d^2,Nh);
for k=1:Nh
    Ck(:,:,k) = kraus2choi(expm(-1i*theta_k(k)*Sz));    % Choi operator of the channel
end

score_adaptive = run_adaptive_monte_carlo(p_initial, theta_k, Ck, No, k_copies, n_monte_carlo);     % Run the MC simulation
fprintf('Adaptive score: %.6f\n', score_adaptive);      % score

end

function score_adaptive = run_adaptive_monte_carlo(p_updated, theta_k, Ck, No, k_copies, n_monte_carlo)

d = 3;

Sz = eye(d);

for i=1:d
    Sz(i,i) = (i-1);
end

Nh = length(p_updated);

final_scores = zeros(n_monte_carlo, 1);

% --- Start parallel pool if needed
pool = gcp('nocreate');
if isempty(pool)
    parpool;  % uses default number of workers
end

parfor mc = 1:n_monte_carlo
    % Give each MC run its own substream (reproducible + independent)
    s = RandStream('Threefry','Seed','shuffle');
    s.Substream = mc;
    RandStream.setGlobalStream(s);

    final_scores(mc) = single_mc_run(p_updated, theta_k, Ck, No, k_copies, d, Nh, Sz, mc, n_monte_carlo);
end

score_adaptive = mean(final_scores); % average over all MC samples, as we are randomly sampling the true parameter

end

function final_score = single_mc_run(p_updated, theta_k, Ck, No, k_copies, d, Nh, Sz, mc, n_monte_carlo)

theta_true = sample_true_parameter(p_updated, theta_k);     % random sample of the true parameter

p_current = p_updated;
theta_i_prev = [];
for copy = 1:k_copies
    if copy == 1
        estimators = 0:2*pi/(No):2*pi;
        theta_i = estimators(1:No);         % estimator distribution
    else
        theta_i = theta_i_prev;
    end

    Xi = zeros(d^2,d^2,No);
    r = zeros(No,Nh);
    for i=1:No
        for k=1:Nh
            r(i,k) = cos((theta_k(k)-theta_i(i))/2)^2;          % reward function
            Xi(:,:,i) = Xi(:,:,i) + p_current(k)*r(i,k)*Ck(:,:,k);      % computing the Xi's
        end
    end

    [T_adaptive, current_score, ~] = testeroptimization_sdp_kcopy_seesaw(Xi,[d d],1,2,1);       % SDP optimization to get the optimal tester and score

    % SEESAW algorithm (takes as input the optimal tester and optimizes over estimators)

    gap = 1;
    precision = 10^(-6);
    rounds = 0;
    old_score = current_score;
    flag_value = 0;

    while gap>precision

        rounds = rounds + 1;

        [estimators] = estimator_optimization(p_current,T_adaptive,Ck,theta_k);     % optimization over the estimator

        theta_i = estimators;
        Xi = zeros(d^2,d^2,No);
        r = zeros(No,Nh);
        for i=1:No
            for k=1:Nh
                r(i,k) = cos((theta_k(k)-theta_i(i))/2)^2;
                Xi(:,:,i) = Xi(:,:,i) + p_current(k)*r(i,k)*Ck(:,:,k);
            end
        end

        [T_temp,score_temp,flag] = testeroptimization_sdp_kcopy_seesaw(Xi,[d d],1,2,1);     % SDP optimization to get the optimal tester and score

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

    C_true = kraus2choi(expm(-1i*theta_true*Sz));       % Choi operator of the true parameter (randomly sampled at the beginning)
    probs = zeros(No, 1);
    for i = 1:No
        probs(i) = real(trace(T_adaptive(:,:,i) * C_true));         % Simulate measurement with true parameter
    end
    probs = probs / sum(probs);

    % Sample random measurement outcome
    cumulative_probs = cumsum(probs);
    rand_val = rand();
    outcome_idx = find(cumulative_probs >= rand_val, 1, 'first');

    if copy <= k_copies
        likelihood = zeros(Nh, 1);
        for k = 1:Nh
            likelihood(k) = real(trace(T_adaptive(:,:,outcome_idx) * Ck(:,:,k)));
        end
        p_current = likelihood .* p_current;        % Update prior using Bayes rule
        p_current = p_current / sum(p_current);
    end

    if copy == k_copies % for the last copy, store the tester and the estimator function Xi
        T_final = T_adaptive;
        Xi_final = zeros(d^2,d^2,No);
        r_final = zeros(No,Nh);
        for i=1:No
            for k=1:Nh
                %r_final(i,k) = ((theta_k(k)-theta_i(i)))^2; % reward MSE
                r_final(i,k) = cos((theta_k(k)-theta_i(i))/2)^2;
                Xi_final(:,:,i) = Xi_final(:,:,i) + p_current(k)*r_final(i,k)*Ck(:,:,k);
            end
        end
    end

end

final_score = 0;
for i = 1:No
    final_score = final_score + real(trace(T_final(:,:,i) * Xi_final(:,:,i)));
end
%final_scores(mc) = final_score;         % Calculate final score

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
% returns the optimal estimators given the tester T 
Nh = max(size(p));
No = size(T,3);

post = zeros(Nh,No);
avg_sin = zeros(No,1);
avg_cos = zeros(No,1);
estimators = zeros(No,1);

for k = 1:Nh
    for i = 1:No
        post(k,i) = p(k)*real(trace(T(:,:,i)*Ck(:,:,k)));
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

function p = prior_p0(h, hmin, hmax, alpha)     % gives the prior distribution in Eq. ()
    % h: vector of points where the prior is evaluated
    % hmin, hmax: lower and upper bound of the interval
    % alpha: shape parameter (larger alpha -> more peaked prior)

    norm_const = (hmax - hmin) * (exp(alpha/2) * besseli(0, alpha/2) - 1);      % normalization constantv

    arg = pi * (h - hmin) / (hmax - hmin);
    num = exp(alpha * (sin(arg).^2)) - 1;

    p = num ./ norm_const;  % unnormalized prior
    p = p ./ sum(p);        % normalization condition

end
