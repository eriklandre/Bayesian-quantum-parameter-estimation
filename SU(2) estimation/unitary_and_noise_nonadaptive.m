function score_nonadaptive = unitary_and_noise_nonadaptive(no, k_copies, n_monte_carlo, p)

d = 2;
[X,Y,Z] = make_paulis;

nh = 20;
Nh = nh^3;

discretization = -pi:2*pi/(nh):pi;
theta_k = zeros(nh,3);
theta_k(:,1) = discretization(1:nh);
theta_k(:,2) = discretization(1:nh);
theta_k(:,3) = discretization(1:nh);

CkU = zeros(d^2,d^2,Nh);   % unitary-only Choi (reward)
CkN = zeros(d^2,d^2,Nh);   % noisy Choi (measurement)
p_initial = zeros(Nh,1);
qk        = zeros(4,Nh);

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
                p_initial(k) = J;

                U = expm(-1i*(thx*X + thy*Y + thz*Z));

                % amplitude damping after unitary
                A0 = diag([1, sqrt(1-p)]);
                A1 = [0, sqrt(p); 0, 0];

                K_unitary = U;                   
                K_total   = cat(3, A0*U, A1*U);

                CkU(:,:,k) = kraus2choi(K_unitary);
                CkN(:,:,k) = kraus2choi(K_total);
                qk(:,k)    = theta_to_quat(th(:));
            else
                p_initial(k) = 0;
            end
        end
    end
end

p_initial = p_initial / sum(p_initial);

[score_nonadaptive, results] = run_nonadaptive_tester_monte_carlo_parallel(p_initial, theta_k, CkU, CkN, qk, no, k_copies, n_monte_carlo, p);

save("su2_concatenation_nonadaptive_tester_MCresults.mat","results");
fprintf('Non-adaptive-tester score: %.6f\n', score_nonadaptive);

end

function [score_nonadaptive, results] = run_nonadaptive_tester_monte_carlo_parallel(p_prior, theta_k, CkU, CkN, qk, no, k_copies, n_monte_carlo, p)

d = 2;
[X,Y,Z] = make_paulis;

Nh = length(p_prior);
No = no^3;

final_scores = zeros(n_monte_carlo,1);
estimator_history = zeros(k_copies, n_monte_carlo, 3);

CkU_const     = parallel.pool.Constant(CkU);
CkN_const     = parallel.pool.Constant(CkN);
theta_k_const = parallel.pool.Constant(theta_k);
p0_const      = parallel.pool.Constant(p_prior);
qk_const      = parallel.pool.Constant(qk);

estimators = -pi:2*pi/(no):pi;
theta_grid = zeros(no,3);
theta_grid(:,1) = estimators(1:no);
theta_grid(:,2) = estimators(1:no);
theta_grid(:,3) = estimators(1:no);

Ci_init = zeros(d^2,d^2,No);
idx = 0;
for ix = 1:no
    for iy = 1:no
        for iz = 1:no
            idx = idx + 1;
            U = expm(-1i*(theta_grid(ix,1)*X + theta_grid(iy,2)*Y + theta_grid(iz,3)*Z));
            Ci_init(:,:,idx) = kraus2choi(U);
        end
    end
end

Xi = zeros(d^2,d^2,No);
for ii = 1:No
    for kk = 1:Nh
        r_ik = (1/(d^2))*real(trace(Ci_init(:,:,ii)*CkU(:,:,kk)));
        Xi(:,:,ii) = Xi(:,:,ii) + p_prior(kk)*r_ik*CkN(:,:,kk);
    end
end

[T_fixed, ~, ~] = testeroptimization_sdp_kcopy_seesaw(Xi,[d d],1,1,1);
T_fixed_const = parallel.pool.Constant(T_fixed);

parfor mc = 1:n_monte_carlo

    CkN_loc     = CkN_const.Value;
    theta_k_loc = theta_k_const.Value;
    qk_loc      = qk_const.Value;
    p_current = p0_const.Value;
    T_fixed = T_fixed_const.Value;

    theta_true = sample_true_parameter(p_current, theta_k_loc);

    U_true = expm(-1i*(theta_true(1)*X + theta_true(2)*Y + theta_true(3)*Z));

    A0 = diag([1, sqrt(1-p)]);
    A1 = [0, sqrt(p); 0, 0];

    C_trueN = kraus2choi(cat(3, A0*U_true, A1*U_true)); 
    C_trueU = kraus2choi(U_true);                      

    for copy = 1:k_copies
        % Sample outcome using FIXED tester + true noisy channel
        probs = zeros(No,1);
        for ii = 1:No
            probs(ii) = real(trace(T_fixed(:,:,ii) * C_trueN));
        end
        probs = probs / sum(probs);
        outcome_idx = find(cumsum(probs) >= rand(), 1, 'first');

        theta_i_round = estimator_update_closedform(p_current, T_fixed, CkN_loc, qk_loc);
        theta_hat = theta_i_round(outcome_idx,:);
        estimator_history(copy, mc, :) = theta_hat;
        % Update posterior (unless first copy uses prior only)
        likelihood = zeros(Nh,1);
        for kk = 1:Nh
            likelihood(kk) = real(trace(T_fixed(:,:,outcome_idx) * CkN_loc(:,:,kk)));
        end
        p_current = likelihood .* p_current;
        p_current = p_current / sum(p_current);

        if copy == k_copies
            Ci_round = zeros(d^2,d^2,No);
            for i = 1:No
                th = theta_i_round(i,:);
                U  = expm(-1i*( th(1)*X + th(2)*Y + th(3)*Z ));
                Ci_round(:,:,i) = kraus2choi(U);
            end
            final_scores(mc) = (1/(d^2)) * real(trace(Ci_round(:,:,outcome_idx) * C_trueU));
        end
    end
end

score_nonadaptive = mean(final_scores);

results.final_scores      = final_scores;
results.estimator_history = estimator_history;
results.score_nonadaptive = score_nonadaptive;
results.No                = No;
results.k_copies          = k_copies;
results.n_monte_carlo     = n_monte_carlo;
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
    F = F + [T_i(:,:,i)>=0];
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

solution = optimize(F,-minmax*score,ops);

T_i   = double(T_i);
score = double(score);

end


function theta_true = sample_true_parameter(p, theta_k)
cumulative_p = cumsum(p);
u = rand();
true_idx = find(cumulative_p >= u, 1, 'first');

nh = size(theta_k, 1);
[kx, ky, kz] = ind2sub([nh, nh, nh], true_idx);

theta_true = [theta_k(kx,1), theta_k(ky,2), theta_k(kz,3)];
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

C = zeros(d_in*d_out,d_in*d_out);

for i = 1:nkraus
    C = C + kron(eye(d_in),K(:,:,i)) * psi * (psi') * kron(eye(d_in),K(:,:,i)');
end

end


function theta_est = estimator_update_closedform(p, T, Ck, qk)

No = size(T,3);
Nh = length(p);

theta_est = zeros(No,3);

for i=1:No
    lik = zeros(Nh,1);

    % Pr(i|k) = Tr[T_i * Ck]
    Ti = T(:,:,i);
    for k=1:Nh
        lik(k) = real(trace(Ti * Ck(:,:,k)));
    end

    w = p(:) .* lik;
    w = w / sum(w);

    K = zeros(4,4);
    for k=1:Nh
        q = qk(:,k);
        K = K + w(k) * (q*q.');
    end
    K = (K + K.')/2;

    [V,D] = eig(K);
    [~,idx] = max(real(diag(D)));
    qhat = real(V(:,idx));
    qhat = qhat / norm(qhat);

    theta_est(i,:) = quat_to_theta(qhat).';
end
end


function q = theta_to_quat(theta)
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
