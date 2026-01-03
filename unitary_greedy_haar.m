function score_adaptive = unitary_greedy_haar(no, k_copies, n_monte_carlo)
% no: number of outcomes of the measurement
% k_copies: number of uses of the channel to use in the Greedy algorithm
% n_monte_carlo: number of Monte Carlo samples

d = 2;

[X,Y,Z] = make_paulis;      % generate Pauli matrices

nh = 10; 
Nh = nh^3;      % number of hypotheses

discretization = -pi:2*pi/(nh):pi;  % true parameter discretization

theta_k = zeros(nh,3);

theta_k(:,1) = discretization(1:nh); % theta_k x
theta_k(:,2) = discretization(1:nh); % theta_k y
theta_k(:,3) = discretization(1:nh); % theta_k z      

Ck = zeros(d^2,d^2,Nh);
p_initial  = zeros(Nh,1);
k = 0;
for kx = 1:nh
    for ky = 1:nh
        for kz = 1:nh
        k = k + 1;

        thx = theta_k(kx,1);
        thy = theta_k(ky,2);
        thz = theta_k(kz,3);
        r   = sqrt(thx^2 + thy^2 + thz^2);

        if r <= pi      % Only keep r <= pi (boundary has zero measure)
            if r < 1e-12
            J = 1/(2*pi^2);     % limit r->0 sin(r)/r = 1
            else
            J = (1/(2*pi^2)) * (sin(r)/r)^2;        % Haar density J(th) = (1/(2*pi^2)) * (sin r / r)^2 
            end
            p_initial(k) = J;

            U = expm(1i*( thx*X + thy*Y + thz*Z ));
            Ck(:,:,k) = kraus2choi(U);      % Choi of the true parameter theta_k
        else
            p_initial(k) = 0;
        end
        end
    end
end

p_initial = p_initial / sum(p_initial);     % normalization condition

score_adaptive = run_adaptive_monte_carlo(p_initial, theta_k, Ck, no, k_copies, n_monte_carlo);     % Run the MC simulation
fprintf('Adaptive score: %.6f\n', score_adaptive);      % score 

end

function score_adaptive = run_adaptive_monte_carlo(p_updated, theta_k, Ck, no, k_copies, n_monte_carlo)

d = 2;
[X,Y,Z] = make_paulis;

No = no^3;
Nh = length(p_updated);

final_scores = zeros(n_monte_carlo, 1);

for mc = 1:n_monte_carlo
    fprintf('Monte Carlo run %d of %d...\n', mc, n_monte_carlo);
    theta_true = sample_true_parameter(p_updated, theta_k);     % random sample of the true parameter

    p_current = p_updated;
    theta_i_prev = [];
    for copy = 1:k_copies
        if copy == 1
            estimators = -pi:2*pi/(no):pi;      % estimator distribution
            theta_i = zeros(no,3);
            theta_i(:,1) = estimators(1:no);
            theta_i(:,2) = estimators(1:no);
            theta_i(:,3) = estimators(1:no);
        else
            fprintf('[MC %d] Copy 2 START - estimators:\n', mc);
            theta_i = theta_i_prev;
        end

        No = no^3; % Number of outcomes of the measurement
        
        Ci = zeros(d^2,d^2,No);
        i=0;
        for ix=1:no
            for iy=1:no
                for iz=1:no
                    i = i+1;
                    Ci(:,:,i) = kraus2choi(expm(1i*(theta_i(ix,1)*X+theta_i(iy,2)*Y+theta_i(iz,3)*Z)));         % Choi operators of the estimators
                end 
            end
        end
        
        Xi = zeros(d^2,d^2,No);
        r = zeros(No,Nh);
        for i=1:No
            for k=1:Nh
                r(i,k) = (1/(d^2))*real(trace(Ci(:,:,i)*Ck(:,:,k)));        % reward function
                Xi(:,:,i) = Xi(:,:,i) + p_current(k)*r(i,k)*Ck(:,:,k);      % computing the Xi's
            end
        end
        
        [T_adaptive, current_score, ~] = testeroptimization_sdp_kcopy_seesaw(Xi,[d d],1,2,1);       % SDP optimization to get the optimal tester and score

        % SEESAW algorithm (takes as input the optimal tester and optimizes over estimators)

        gap        = 1;
        precision  = 1e-6;
        rounds     = 0;
        old_score  = current_score;
        flag_value = 0;

        while gap > precision
            rounds = rounds + 1;

            [estimators, ~] = estimator_optimization(p_current, T_adaptive, Ck, theta_i);       % optimization over the estimators
            
            theta_i = estimators;

            Ci = zeros(d^2,d^2,No);
            i = 0;
            for ix=1:no
                for iy=1:no
                    for iz=1:no
                        i = i+1;
                        Ci(:,:,i) = kraus2choi(expm(1i*(theta_i(ix,1)*X + theta_i(iy,2)*Y + theta_i(iz,3)*Z)));
                    end
                end
            end

            Xi = zeros(d^2,d^2,No);
            r = zeros(No,Nh);
            for i=1:No
                for k=1:Nh
                    r(i,k) = (1/(d^2))*real(trace(Ci(:,:,i)*Ck(:,:,k))); % reward
                    Xi(:,:,i) = Xi(:,:,i) + p_current(k)*r(i,k)*Ck(:,:,k);
                end
            end

            [T_temp, score_temp, flag] = testeroptimization_sdp_kcopy_seesaw(Xi,[d d],1,2,1);       % SDP optimization to get the optimal tester and score

            if flag.problem~=0
                sdp_problem = flag.problem
                info        = flag.info
                flag_value = 1;
                %pause
            end

            gap       = abs(old_score - score_temp);
            old_score = score_temp;
        end

        T_adaptive   = T_temp;
        current_score = score_temp;
        theta_i_prev = theta_i;
        
        C_true = kraus2choi(expm(1i*(theta_true(1)*X + theta_true(2)*Y + theta_true(3)*Z)));        % Choi operator of the true parameter (randomly sampled at the beginning)
        probs = zeros(No, 1);
        for i = 1:No
            probs(i) = real(trace(T_adaptive(:,:,i) * C_true));     % Simulate measurement with true parameter
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
                    r_final(i,k) = (1/(d^2))*real(trace(Ci(:,:,i)*Ck(:,:,k)));
                    Xi_final(:,:,i) = Xi_final(:,:,i) + p_current(k)*r_final(i,k)*Ck(:,:,k);
                end
            end
        end

    end
    
    final_score = 0;
    for i = 1:No
        final_score = final_score + real(trace(T_final(:,:,i) * Xi_final(:,:,i)));
    end
    final_scores(mc) = final_score;     % Calculate final score
end

score_adaptive = mean(final_scores); % average over all MC samples, as we are randomly sampling the true parameter


end

function theta_true = sample_true_parameter(p, theta_k)
    % p: prior distribution
    % theta_k: grid of hypotheses

    cumulative_p = cumsum(p);
    rand_val = rand();
    true_idx = find(cumulative_p >= rand_val, 1, 'first');
    nh = size(theta_k, 1);
    [kx, ky, kz] = ind2sub([nh, nh, nh], true_idx);
    theta_true = [theta_k(kx,1), theta_k(ky,2), theta_k(kz,3)];
end

function [estimators,score] = estimator_optimization(p,T,Ck,theta_i)

options = optimset('MaxFunEvals',50,'MaxIter',50,'TolX',1e-5,'TolFun',1e-5);

x0 = theta_i;

[estimators,fval] = fminsearch(@(x)objectivefcn(x,p,T,Ck),x0,options);

score = -fval;

end

function score = objectivefcn(theta_i,p,T,Ck)

[X,Y,Z] = make_paulis;

d = 2;

No = size(T,3);
Nh = max(size(p));
no = size(theta_i,1);

Ci = zeros(d^2,d^2,No);
i=0;
for ix=1:no
    for iy=1:no
        for iz=1:no
            i = i+1;
            Ci(:,:,i) = kraus2choi(expm(1i*(theta_i(ix,1)*X+theta_i(iy,2)*Y+theta_i(iz,3)*Z)));
        end
    end
end

r = zeros(No,Nh);
score = 0;
for i=1:No
    for k=1:Nh
        r(i,k) = (1/(d^2))*real(trace(Ci(:,:,i)*Ck(:,:,k)));
        score = score + p(k)*r(i,k)*real(trace(T(:,:,i)*Ck(:,:,k)));
    end
end

score = -score; % to maximize

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

        