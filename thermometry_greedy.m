function score_adaptive = thermometry_greedy(No, k_copies, n_monte_carlo, t)
% No: number of outcomes of the measurement
% k_copies: number of uses of the channel
% n_monte_carlo: number of Monte Carlo samples
% t: index of the time point

d = 2;

Tmin = 0.1;     % minimum temperature
Tmax = 2;       % maximum temperature
Nh   = 2500;     % number of hypotheses
Nt   = 100;     % number of time steps

time = 0:(1/(Nt-1)):1;   % time vector
time_value = time(t);

theta_k(1:Nh) = Tmin:(Tmax-Tmin)/(Nh-1):Tmax;   % hypothesis grid 

p_initial(1:Nh,1) = 1/Nh;            % uniform prior

% Precompute Choi operators
Ck = zeros(d^2,d^2,Nh);
for k=1:Nh
    Ck(:,:,k) = ChoiOperatorThermo(theta_k(k),0.1,2,time(t));       % Choi of the true parameter theta_k
end

% Run MC simulation in parallel
[score_adaptive, results] = run_adaptive_monte_carlo_parallel(p_initial, theta_k, Ck, No, k_copies, n_monte_carlo, time_value);
folder = 'results_files';
if ~exist(folder,'dir'); 
    mkdir(folder); 
end
save(fullfile(folder, sprintf('results_t=%d.mat', t)), 'results');

end

function [score_adaptive, results] = run_adaptive_monte_carlo_parallel(p_initial, theta_k, Ck, No, k_copies, n_monte_carlo, time_value)

d = 2;
Nh = length(p_initial);

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

    final_scores(mc) = single_mc_run(p_initial, theta_k, Ck, No, k_copies, time_value, d, Nh);
end

score_adaptive = mean(final_scores);

% Package results
results.final_scores   = final_scores;
results.score_adaptive = score_adaptive;
results.No             = No;
results.k_copies       = k_copies;
results.n_monte_carlo  = n_monte_carlo;
results.time_value     = time_value;
results.theta_grid     = theta_k;

end

function score_run = single_mc_run(p_initial, theta_k, Ck, No, k_copies, time_value, d, Nh)
% One independent MC run of the greedy adaptive protocol

Tmin = min(theta_k);
Tmax = max(theta_k);

% sample true parameter
theta_true = sample_true_parameter(p_initial, theta_k);

p_current = p_initial;
theta_i_prev = [];

for copy = 1:k_copies

    % Initial estimator grid
    if copy == 1
        theta_i(1:No) = Tmin:(Tmax-Tmin)/(No-1):Tmax;
    else
        theta_i = theta_i_prev;
    end

    % Build Xi
    Xi = zeros(d^2,d^2,No);
    r = zeros(No,Nh);
    for i=1:No
        for k=1:Nh
            r(i,k) = ((theta_k(k)-theta_i(i))/theta_k(k))^2; % cost function (relative error)
            Xi(:,:,i) = Xi(:,:,i) + p_current(k)*r(i,k)*Ck(:,:,k);      % computing the Xi's
        end
    end

    % Optimize tester
    [T_adaptive, current_score, ~] = testeroptimization_sdp_kcopy_seesaw(Xi,[d d],1,2,-1);

    % Seesaw: optimize estimators given the tester
    T_temp = T_adaptive;
    gap = 1;
    precision = 1e-6;
    old_score = current_score;

    while gap > precision

        theta_i = estimator_optimization(p_current, T_temp, Ck, theta_k);
        Xi = zeros(d^2,d^2,No);
        r = zeros(No,Nh);
        for i=1:No
            for k=1:Nh
                r(i,k) = ((theta_k(k)-theta_i(i))/theta_k(k))^2; % relative error
                Xi(:,:,i) = Xi(:,:,i) + p_current(k)*r(i,k)*Ck(:,:,k);
            end
        end

        [T_temp, score_temp, ~] = testeroptimization_sdp_kcopy_seesaw(Xi,[d d],1,2,-1);

        gap = abs(old_score - score_temp);
        old_score = score_temp;
        
    end

    T_adaptive = T_temp;
    theta_i_prev = theta_i;

    % simulate measurement outcome with true parameter
    C_true = ChoiOperatorThermo(theta_true, 0.1, 2, time_value);

    probs = zeros(No, 1);
    for i = 1:No
        probs(i) = real(trace(T_adaptive(:,:,i) * C_true));
    end
    probs = probs / sum(probs);

    outcome_idx = sample_discrete(probs);

    if copy < k_copies
        % Bayes update
        likelihood = zeros(Nh, 1);
        for k = 1:Nh
            likelihood(k) = real(trace(T_adaptive(:,:,outcome_idx) * Ck(:,:,k)));
        end
        p_current = likelihood .* p_current;
        p_current = p_current / sum(p_current);
    else
        % LAST COPY: realized cost for THIS MC run
        theta_hat = theta_i(outcome_idx);
        score_run = ((theta_true - theta_hat) / theta_true)^2;
    end

end

end

function idx = sample_discrete(p)
% samples an index from probability vector p
c = cumsum(p(:));
u = rand();
idx = find(c >= u, 1, 'first');
end


function theta_true = sample_true_parameter(p, theta_k)
% samples theta_true from the discrete prior p
c = cumsum(p(:) / sum(p));
u = rand();
idx = find(c >= u, 1, 'first');
theta_true = theta_k(idx);
end


function [estimators] = estimator_optimization(p,T,Ck,theta_k)
% Given tester T, compute optimal estimators
Nh = length(p);
No = size(T,3);

post = zeros(Nh, No);

for k = 1:Nh
    for i = 1:No
        post(k,i) = p(k) * real(trace(T(:,:,i) * Ck(:,:,k)));
    end
end
post = post ./ sum(post,1);

estimators = zeros(No,1);

for i = 1:No
    num_1 = 0;
    den_1 = 0;
    for k = 1:Nh
        num_1 = num_1 + post(k,i) * (1/theta_k(k));
        den_1 = den_1 + post(k,i) * (1/(theta_k(k)^2));
    end
    estimators(i) = num_1 / den_1;
end

end


function J = ChoiOperatorThermo(T,eps,g,t)
% Choi operator of the thermometry channel at temperature T
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

J = 2 * J;
J = Swap(J);

end
