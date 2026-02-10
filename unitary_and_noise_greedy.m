function score_adaptive = unitary_and_noise_greedy(no, k_copies, n_monte_carlo, pAD)
% no: number of outcomes of the measurement
% k_copies: number of uses of the channel
% n_monte_carlo: number of Monte Carlo samples
% pAD: amplitude damping parameter

d = 2;
[X,Y,Z] = make_paulis;

nh = 20;
Nh = nh^3;

discretization = -pi:2*pi/(nh):pi;
theta_k = zeros(nh,3); % hypotheses grid
theta_k(:,1) = discretization(1:nh);
theta_k(:,2) = discretization(1:nh);
theta_k(:,3) = discretization(1:nh);

CkU = zeros(d^2,d^2,Nh);
CkN = zeros(d^2,d^2,Nh);
p0  = zeros(Nh,1);
qk  = zeros(4,Nh);

k = 0;
for kx = 1:nh
    for ky = 1:nh
        for kz = 1:nh
            k = k + 1;

            thx = theta_k(kx,1);
            thy = theta_k(ky,2);
            thz = theta_k(kz,3);
            th  = [thx, thy, thz];
            r   = norm(th);

            if r <= pi
                % Haar density in fundamental ball
                if r < 1e-12
                    J = 1/(2*pi^2);
                else
                    J = (1/(2*pi^2)) * (sin(r)/r)^2;
                end
                p0(k) = J;

                U = expm(-1i*(thx*X + thy*Y + thz*Z));

                % amplitude damping after unitary
                A0 = diag([1, sqrt(1-pAD)]);
                A1 = [0, sqrt(pAD); 0, 0];

                K_unitary = U;                   % 2x2 (1 Kraus)
                K_total   = cat(3, A0*U, A1*U);   % 2x2x2

                CkU(:,:,k) = kraus2choi(K_unitary);
                CkN(:,:,k) = kraus2choi(K_total);
                qk(:,k)    = theta_to_quat(th(:));
            else
                p0(k) = 0;
            end
        end
    end
end

p0 = p0 / sum(p0);

% Precompute (T, theta, Ci) for the first copy
[T1, theta1, Ci1] = precompute_copy1_seesaw(no, p0, CkU, CkN, qk);

% Run adaptive MC (parallel)
[score_adaptive, results] = run_adaptive_monte_carlo_parallel(p0, theta_k, CkU, CkN, qk, no, k_copies, n_monte_carlo, pAD, T1, theta1, Ci1);

end

function [T1, theta1, Ci1] = precompute_copy1_seesaw(no, p0, CkU, CkN, qk)

d = 2;
[X,Y,Z] = make_paulis;

Nh = length(p0);
No = no^3;

estimators = -pi:2*pi/(no):pi;
theta_grid = zeros(no,3); % estimators grid
theta_grid(:,1) = estimators(1:no);
theta_grid(:,2) = estimators(1:no);
theta_grid(:,3) = estimators(1:no);

% Build Ci from the separable grid (No outcomes)
Ci = zeros(d^2,d^2,No);
idx = 0;
for ix = 1:no
    for iy = 1:no
        for iz = 1:no
            idx = idx + 1;
            U = expm(-1i*(theta_grid(ix,1)*X + theta_grid(iy,2)*Y + theta_grid(iz,3)*Z));
            Ci(:,:,idx) = kraus2choi(U);
        end
    end
end

% Seesaw until convergence
gap = 1;
precision = 1e-6;
rounds = 0;

% Build Xi and solve initial tester
Xi = build_Xi_from_estimators(p0, Ci, CkU, CkN);
[T_temp, score_old, ~] = testeroptimization_sdp_kcopy_seesaw(Xi, [d d], 1, 1, 1);

while gap > precision
    rounds = rounds + 1;

    % Closed-form estimator update 
    theta_list = estimator_update_closedform(p0, T_temp, CkN, qk); % No x 3

    Ci = zeros(d^2,d^2,No);
    for i = 1:No
        th = theta_list(i,:);
        U  = expm(-1i*(th(1)*X + th(2)*Y + th(3)*Z));
        Ci(:,:,i) = kraus2choi(U);
    end

    % Rebuild Xi and optimize tester
    Xi = build_Xi_from_estimators(p0, Ci, CkU, CkN);
    [T_new, score_new, flag] = testeroptimization_sdp_kcopy_seesaw(Xi, [d d], 1, 1, 1);

    gap = abs(score_old - score_new);
    score_old = score_new;
    T_temp = T_new;
end

T1     = T_temp;
theta1 = theta_list;
Ci1    = Ci;

end

function [score_adaptive, results] = run_adaptive_monte_carlo_parallel(p0, theta_k, CkU, CkN, qk, no, k_copies, n_monte_carlo, pAD, T1, theta1, Ci1)

d = 2;
[X,Y,Z] = make_paulis;

Nh = length(p0);
No = no^3;

final_scores = zeros(n_monte_carlo,1);

% Wrap arrays to reduce broadcast overhead in parfor
CkU_const     = parallel.pool.Constant(CkU);
CkN_const     = parallel.pool.Constant(CkN);
theta_k_const = parallel.pool.Constant(theta_k);
p0_const      = parallel.pool.Constant(p0);
qk_const      = parallel.pool.Constant(qk);

T1_const      = parallel.pool.Constant(T1);
theta1_const  = parallel.pool.Constant(theta1);
Ci1_const     = parallel.pool.Constant(Ci1);

parfor mc = 1:n_monte_carlo

    CkU_loc = CkU_const.Value;
    CkN_loc = CkN_const.Value;
    theta_k_loc = theta_k_const.Value;
    p_current = p0_const.Value;
    qk_loc = qk_const.Value;

    T1_loc = T1_const.Value;
    theta1_loc = theta1_const.Value;
    Ci1_loc = Ci1_const.Value;

    % Sample true theta from prior (discrete)
    theta_true = sample_true_parameter(p_current, theta_k_loc);

    theta_i_prev = [];   % will be No×3 list after first copy

    for copy = 1:k_copies
        % Use precomputed copy-1 solution
        if copy == 1
            T_adaptive  = T1_loc;
            theta_i     = theta1_loc;   
            Ci          = Ci1_loc;      
        else
            theta_i = theta_i_prev;

            % Build Ci from No×3 list
            Ci = zeros(d^2,d^2,No);
            for ii = 1:No
                th = theta_i(ii,:);
                U  = expm(-1i*(th(1)*X + th(2)*Y + th(3)*Z));
                Ci(:,:,ii) = kraus2choi(U);
            end

            % Build Xi from current posterior and solve tester + seesaw
            Xi = build_Xi_from_estimators(p_current, Ci, CkU_loc, CkN_loc);
            [T_temp, score_temp, ~] = testeroptimization_sdp_kcopy_seesaw(Xi, [d d], 1, 1, 1);

            gap = 1;
            precision = 1e-6;
            old_score = score_temp;

            while gap > precision
                theta_i = estimator_update_closedform(p_current, T_temp, CkN_loc, qk_loc);
                Ci = zeros(d^2,d^2,No);
                for ii = 1:No
                    th = theta_i(ii,:);
                    U  = expm(-1i*(th(1)*X + th(2)*Y + th(3)*Z));
                    Ci(:,:,ii) = kraus2choi(U);
                end

                % Xi from posterior
                Xi = build_Xi_from_estimators(p_current, Ci, CkU_loc, CkN_loc);

                [T_new, score_new, flag] = testeroptimization_sdp_kcopy_seesaw(Xi, [d d], 1, 1, 1);

                gap = abs(old_score - score_new);
                old_score = score_new;
                T_temp = T_new;
            end

            T_adaptive = T_temp;
        end

        % True channels (noisy for measurement, unitary for reward)
        U_true = expm(-1i*(theta_true(1)*X + theta_true(2)*Y + theta_true(3)*Z));
        A0 = diag([1, sqrt(1-pAD)]);
        A1 = [0, sqrt(pAD); 0, 0];

        C_trueN = kraus2choi(cat(3, A0*U_true, A1*U_true));
        C_trueU = kraus2choi(U_true);

        % Sample outcome from tester + true noisy channel
        probs = zeros(No,1);
        for ii = 1:No
            probs(ii) = real(trace(T_adaptive(:,:,ii) * C_trueN));
        end
        probs = max(probs, 0);
        sp = sum(probs);
        if sp < 1e-15
            probs = ones(No,1)/No;
        else
            probs = probs / sp;
        end
        outcome_idx = find(cumsum(probs) >= rand(), 1, 'first');

        % Bayesian update (except last copy)
        if copy < k_copies
            likelihood = zeros(Nh,1);
            for k = 1:Nh
                likelihood(k) = real(trace(T_adaptive(:,:,outcome_idx) * CkN_loc(:,:,k)));
            end
            likelihood = max(likelihood, 0);

            p_current = p_current .* likelihood;
            sp = sum(p_current);
            if sp < 1e-15
                % fallback: keep prior (or keep previous p_current)
                p_current = p0_const.Value;
            else
                p_current = p_current / sp;
            end

            theta_i_prev = theta_i; 
        else
            % final realized reward (unitary fidelity reward)
            final_scores(mc) = (1/(d^2)) * real(trace(Ci(:,:,outcome_idx) * C_trueU));
        end
    end
end

score_adaptive = mean(final_scores);

results.final_scores   = final_scores;
results.score_adaptive = score_adaptive;
results.No             = No;
results.k_copies       = k_copies;
results.n_monte_carlo  = n_monte_carlo;
end


function Xi = build_Xi_from_estimators(p, Ci, CkU, CkN)
% Build Xi for tester optimization from estimators Ci and posterior p
d = 2;
No = size(Ci,3);
Nh = length(p);

Xi = zeros(d^2,d^2,No);

for ii = 1:No
    acc = zeros(d^2,d^2);
    Ci_ii = Ci(:,:,ii);
    for k = 1:Nh
        r_ik = (1/(d^2)) * real(trace(Ci_ii * CkU(:,:,k)));
        acc  = acc + p(k) * r_ik * CkN(:,:,k);
    end
    Xi(:,:,ii) = acc;
end
end

function [T_i,score, solution] = testeroptimization_sdp_kcopy_seesaw(Xk_i,d,k,strategy,minmax)

din  = d(1);
dout = d(2);
No   = size(Xk_i,3);

dvec = repmat([din dout], 1, k);

yalmip('clear');

T_i = sdpvar((din*dout)^k,(din*dout)^k,No,'hermitian','complex');

F = [trace(sum(T_i,3))==dout^k];

score = 0;
for i = 1:No
    F = F + [T_i(:,:,i) >= 0];
    score = score + real(trace(T_i(:,:,i)*Xk_i(:,:,i)));
end

if strategy==1
    F = F + [sum(T_i,3)==ProjParProcess(sum(T_i,3),dvec)];
elseif strategy==2
    F = F + [sum(T_i,3)==ProjSeqProcess(sum(T_i,3),dvec)];
elseif strategy==3
    F = F + [sum(T_i,3)==ProjGenProcess(sum(T_i,3),dvec)];
end

ops = sdpsettings('solver','mosek','verbose',0,'cachesolvers',1);
ops.mosek.MSK_IPAR_NUM_THREADS = 1;

solution = optimize(F, -minmax*score, ops);

T_i   = double(T_i);
score = double(score);

end

function theta_true = sample_true_parameter(p, theta_k)
% Sample true theta from discrete distribution p on the theta_k grid
cumulative_p = cumsum(p);
u = rand();
true_idx = find(cumulative_p >= u, 1, 'first');

nh = size(theta_k, 1);
[kx, ky, kz] = ind2sub([nh, nh, nh], true_idx);

theta_true = [theta_k(kx,1), theta_k(ky,2), theta_k(kz,3)];
end

function theta_est = estimator_update_closedform(p, T, CkN, qk)
% optimization of the estimators given the tester T (closed-form solution)
No = size(T,3);
Nh = length(p);

theta_est = zeros(No,3);

for i = 1:No
    lik = zeros(Nh,1);
    Ti = T(:,:,i);

    for k = 1:Nh
        lik(k) = real(trace(Ti * CkN(:,:,k)));
    end

    lik = max(lik,0);
    w = p(:).*lik;
    sw = sum(w);
    if sw < 1e-15
        w = p(:);
        sw = sum(w);
    end
    w = w/sw;

    K = zeros(4,4);
    for k = 1:Nh
        q = qk(:,k);
        if any(q)
            K = K + w(k) * (q*q.');
        end
    end
    K = (K + K.')/2;

    [V,D] = eig(K);
    [~,idx] = max(real(diag(D)));
    qhat = real(V(:,idx));
    if qhat(1) < 0, qhat = -qhat; end
    qhat = qhat / norm(qhat);
    theta_est(i,:) = quat_to_theta(qhat).';
end
end

function [X,Y,Z] = make_paulis()
X = [0 1; 1 0];
Y = [0 -1i; 1i 0];
Z = [1 0; 0 -1];
end

function C = kraus2choi(K)

d_out  = size(K,1);
d_in   = size(K,2);
nkraus = size(K,3);

psi = zeros(d_in^2,1);
vec = eye(d_in);

for i = 1:d_in
    psi = psi + kron(vec(:,i),vec(:,i));
end

C = zeros(d_in*d_out, d_in*d_out);

for i = 1:nkraus
    C = C + kron(eye(d_in), K(:,:,i)) * psi * (psi') * kron(eye(d_in), K(:,:,i)');
end
end

function q = theta_to_quat(theta)
% Converts a parameter vector to a quaternion
r = norm(theta);
if r < 1e-12
    q = [1;0;0;0];
else
    q0 = cos(r);
    qv = sin(r) * (theta / r);
    q  = [q0; qv(:)];
end
end

function theta = quat_to_theta(q)
% Converts a quaternion to a parameter vector
q = q / norm(q);
q0 = max(min(q(1),1),-1);

r = acos(q0);
s = sin(r);

if s < 1e-12
    theta = [0;0;0];
else
    n = q(2:4) / s;
    theta = r * n;
end
end
