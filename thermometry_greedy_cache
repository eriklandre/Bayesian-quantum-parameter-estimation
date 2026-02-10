function score_adaptive = thermometry_greedy_cache(No, k_copies, n_monte_carlo, t)

d = 2;

% Setting the parameters
Tmin = 0.1;
Tmax = 2.0;
Nh   = 2500;
Nt   = 100;

eps_val = 0.1;
g_val   = 2.0;

time = 0:(1/(Nt-1)):1;
time_value = time(t);

theta_k(1:Nh) = Tmin:(Tmax-Tmin)/(Nh-1):Tmax;   % hypotheses
p_initial(1:Nh,1) = 1/Nh;            % uniform prior

cache_dir = sprintf('thermo_cache/No=%d_k=%d_Nh=%d_t=%.6f_eps=%.6f_g=%.6f', No, k_copies, Nh, time_value, eps_val, g_val); % this creates the cache folder where all the data will be stored

if ~exist(cache_dir,'dir') % if it already exists, we skip creating it
    mkdir(cache_dir);
end

Ck = zeros(d^2, d^2, Nh);
for k = 1:Nh
    Ck(:,:,k) = ChoiOperatorThermo(theta_k(k), eps_val, g_val, time_value);
end

copy1_file = fullfile(cache_dir, 'copy1.mat'); % create the file to store the tester and estimators for the first copy

if exist(copy1_file,'file') % if it exists, just load the data
    S = load(copy1_file, 'T1', 'theta1');
    T1 = S.T1;
    theta1 = S.theta1;
else
    [T1, theta1] = solve_one_round_greedy(p_initial, theta_k, Ck, No); % if it doesn't exist, solve for the first copy tester and estimators and save them
    save(copy1_file, 'T1', 'theta1', '-v7.3');
end

% Precompute / load Copy-2 caches for each first outcome i1

T2_cell     = cell(No,1);
theta2_cell = cell(No,1);
p1_cell     = cell(No,1);

for i1 = 1:No
    branch_file = fullfile(cache_dir, sprintf('copy2_branch_i1=%d.mat', i1)); % create the file to store the tester and estimators for the second copy branch i1

    if exist(branch_file,'file') % if it exists, just load the data
        S = load(branch_file, 'T2', 'theta2', 'p1');
        T2_cell{i1} = S.T2;
        theta2_cell{i1} = S.theta2;
        p1_cell{i1} = S.p1;
        continue;
    end

    p1 = bayes_update(p_initial, T1, i1, Ck); % posterior after seeing outcome i1 from copy-1 tester

    [T2, theta2] = solve_one_round_greedy(p1, theta_k, Ck, No); % greedy solve for copy-2 given this posterior

    save(branch_file, 'T2', 'theta2', 'p1', '-v7.3'); % save the copy-2 branch data for this i1

    T2_cell{i1} = T2;
    theta2_cell{i1} = theta2;
    p1_cell{i1} = p1;
end

% Monte Carlo loop (like in the usual parallelized case)

pool = gcp('nocreate');
if isempty(pool)
    parpool;
end

theta_k_const = parallel.pool.Constant(theta_k);
p0_const      = parallel.pool.Constant(p_initial);
T1_const      = parallel.pool.Constant(T1);
T2_const      = parallel.pool.Constant(T2_cell);
theta2_const  = parallel.pool.Constant(theta2_cell);

final_scores = zeros(n_monte_carlo,1);
traj = zeros(n_monte_carlo, 2);  % store outcomes 

parfor mc = 1:n_monte_carlo
    s = RandStream('Threefry','Seed','shuffle');
    s.Substream = mc;
    RandStream.setGlobalStream(s);

    theta_k_loc = theta_k_const.Value;
    p0 = p0_const.Value;
    T1_loc = T1_const.Value;
    T2_cell_loc = T2_const.Value;
    theta2_cell_loc = theta2_const.Value;

    theta_true = sample_true_parameter(p0, theta_k_loc); % sample true parameter

    C_true = ChoiOperatorThermo(theta_true, eps_val, g_val, time_value); % true channel Choi operator
    
    probs1 = outcome_probs(T1_loc, C_true);
    i1 = sample_discrete(probs1); % copy 1 outcome

    % Copy 2 tester/estimators from cache branch i1
    T2 = T2_cell_loc{i1};
    theta2 = theta2_cell_loc{i1};

    probs2 = outcome_probs(T2, C_true); 
    i2 = sample_discrete(probs2); % copy 2 outcome

    theta_hat = theta2(i2); % estimator from copy 2 for outcome i2

    final_scores(mc) = ((theta_true - theta_hat) / theta_true)^2;
    traj(mc,:) = [i1 i2];
end

score_adaptive = mean(final_scores);

results_dir = 'results_files_thermometry_new'; % Save results to a separate folder 
if ~exist(results_dir,'dir')
    mkdir(results_dir);
end

results.final_scores   = final_scores;
results.score_adaptive = score_adaptive;
results.No             = No;
results.k_copies       = k_copies;
results.n_monte_carlo  = n_monte_carlo;
results.time_value     = time_value;
results.theta_grid     = theta_k;
results.traj_outcomes  = traj;
results.cache_dir      = cache_dir;

outfile = fullfile(results_dir, sprintf('thermometry_greedy_k2_cached_t=%.6f_No=%d_Nh=%d.mat', time_value, No, Nh));
save(outfile, 'results', '-v7.3');

fprintf('Adaptive (cached, k=2) score: %.6f\n', score_adaptive);
fprintf('Cache directory: %s\n', cache_dir);
fprintf('Saved results: %s\n', outfile);

end

function [T_round, theta_round] = solve_one_round_greedy(p, theta_k, Ck, No)
% One greedy round: optimize tester and estimators via seesaw
d  = 2;
Nh = length(p);

Tmin = min(theta_k);
Tmax = max(theta_k);

theta_i = linspace(Tmin, Tmax, No).'; % initial estimator grid

Xi = zeros(d^2,d^2,No);
for i = 1:No
    for k = 1:Nh
        r = ((theta_k(k)-theta_i(i))/theta_k(k))^2;
        Xi(:,:,i) = Xi(:,:,i) + p(k)*r*Ck(:,:,k);
    end
end

[T_temp, score_old, ~] = testeroptimization_sdp_kcopy_seesaw(Xi, [d d], 1, 2, -1); % tester optimization

% Seesaw algorithm
gap = 1;
precision = 1e-6;

while gap > precision
    theta_i = estimator_optimization(p, T_temp, Ck, theta_k); % Estimator optimization

    Xi = zeros(d^2,d^2,No);
    for i = 1:No
        for k = 1:Nh
            r = ((theta_k(k)-theta_i(i))/theta_k(k))^2;
            Xi(:,:,i) = Xi(:,:,i) + p(k)*r*Ck(:,:,k);
        end
    end

    [T_new, score_new, ~] = testeroptimization_sdp_kcopy_seesaw(Xi, [d d], 1, 2, -1); % Tester optimization (seesaw)

    gap = abs(score_old - score_new); % Running until convergence (with predefined precision)
    score_old = score_new;
    T_temp = T_new;
end

T_round = T_temp;
theta_round = theta_i;
end

function p_new = bayes_update(p, T, outcome_idx, Ck)
% Posterior update
Nh = length(p);
lik = zeros(Nh,1);

Tout = T(:,:,outcome_idx);
for k = 1:Nh
    lik(k) = real(trace(Tout * Ck(:,:,k)));
end
lik = max(lik, 0);

p_new = p .* lik;
s = sum(p_new);
if s < 1e-15
    % fallback: keep prior if underflow / numerical collapse
    p_new = p / sum(p);
else
    p_new = p_new / s;
end
end

function probs = outcome_probs(T, C_true)
% Gives the outcome probabilities
No = size(T,3);
probs = zeros(No,1);
for i = 1:No
    probs(i) = real(trace(T(:,:,i) * C_true));
end
probs = max(probs, 0);
s = sum(probs);
if s < 1e-15
    probs = ones(No,1)/No;
else
    probs = probs / s;
end
end

function idx = sample_discrete(p)
c = cumsum(p(:));
u = rand();
idx = find(c >= u, 1, 'first');
end

function theta_true = sample_true_parameter(p, theta_k)
% Randomly samples the true parameter
c = cumsum(p(:) / sum(p));
u = rand();
idx = find(c >= u, 1, 'first');
theta_true = theta_k(idx);
end

function estimators = estimator_optimization(p,T,Ck,theta_k)
% Optimizes the estimators for relative MSE
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
% Choi operator for the thermometry problem
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
