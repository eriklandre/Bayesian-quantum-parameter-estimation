function [T,score] = unitary_and_noise_kcopy(ncopies, strategy, p)

d = 2;

[X,Y,Z] = make_paulis;

no_initial = 7;
no_final   = 7;

score = zeros(no_final-no_initial+1,1);
T     = cell(no_final-no_initial+1,1);

for no=no_initial:no_final
    
    no = no
    nh = 20; 

    No = no^3;
    Nh = nh^3;
   
    estimators = -pi:2*pi/(no):pi;      % uniform distribution of the estimator 
    discretization = -pi:2*pi/(nh):pi;  % uniform discretization of theta 
    
    theta_i = zeros(no,3);
    theta_k = zeros(nh,3);
    
    % will use only first no values, not including 2pi
    theta_i(:,1) = estimators(1:no); % theta_i x
    theta_i(:,2) = estimators(1:no); % theta_i y
    theta_i(:,3) = estimators(1:no); % theta_i z
    
    % will use only first nh values, not including 2pi
    theta_k(:,1) = discretization(1:nh); % theta_k x
    theta_k(:,2) = discretization(1:nh); % theta_k y
    theta_k(:,3) = discretization(1:nh); % theta_k z

    %Ck = zeros(d^2,d^2,Nh);
    CkN = zeros(d^2,d^2,Nh);   % noisy channel choi (what you probe)
    CkU = zeros(d^2,d^2,Nh);   % unitary-only choi (what you score against)
    p_initial  = zeros(Nh,1);
    qk = zeros(4,Nh);
    k = 0;
    for kx = 1:nh
        for ky = 1:nh
            for kz = 1:nh
                k = k + 1;

                thx = theta_k(kx,1);
                thy = theta_k(ky,2);
                thz = theta_k(kz,3);
                r   = sqrt(thx^2 + thy^2 + thz^2);
                th  = [thx, thy, thz];

                % Fundamental ball: only keep r <= pi (boundary has zero measure)
                if r <= pi
                    % Haar density J(th) = (1/(2*pi^2)) * (sin r / r)^2  (with r->0 limit 1)
                    if r < 1e-12
                    J = 1/(2*pi^2);
                    else
                    J = (1/(2*pi^2)) * (sin(r)/r)^2;
                    end
                    p_initial(k) = J;

                    U = expm(-1i*( thx*X + thy*Y + thz*Z ));
                    A0 = diag([1, sqrt(1-p)]);
                    A1 = [0, sqrt(p); 0, 0];
                    KrausT{1,1} = A0*U; 
                    KrausT{2,1} = A1*U;
                    KrausU{1,1} = U;
                    CkU(:,:,k)  = ChoiMatrix(KrausU);   % trace = 1 in your convention
                    CkN(:,:,k) = ChoiMatrix(KrausT);   % normalized convention: trace(Ck)=1
                    qk(:,k)    = theta_to_quat(th(:));
                else
                    % outside fundamental domain -> zero weight (can leave Ck(:,:,k)=0)
                    p_initial(k) = 0;
                end
            end
        end
    end

    % Normalize discrete prior to sum to 1
    p_initial = p_initial / sum(p_initial);
    Ck_tens = zeros(d^(2*ncopies), d^(2*ncopies), Nh);
    for k=1:Nh
        Ck_tens(:,:,k) = Tensor(CkN(:,:,k), ncopies);
    end
    % Ci = zeros(d^2,d^2,No);
    % i=0;
    % for ix=1:no
    %     for iy=1:no
    %         for iz=1:no
    %             i = i+1;
    %             Ci(:,:,i) = kraus2choi(expm(-1i*(theta_i(ix,1)*X+theta_i(iy,2)*Y+theta_i(iz,3)*Z)));
    %         end 
    %     end
    % end
    Ci = zeros(d^2,d^2,No);
    i = 0;
    for ix = 1:no
        for iy = 1:no
            for iz = 1:no
                i = i + 1;
                thx_est = theta_i(ix,1);
                thy_est = theta_i(iy,2);
                thz_est = theta_i(iz,3);
                r   = sqrt(thx_est^2 + thy_est^2 + thz_est^2);
                % Fundamental ball: only keep r <= pi (boundary has zero measure)
                if r <= pi
                    U = expm(-1i*( thx_est*X + thy_est*Y + thz_est*Z ));
                    KrausU{1,1} = U;
                    Ci(:,:,i) = ChoiMatrix(KrausU);
                end
            end
        end
    end
    Xi = zeros(d^(2*ncopies),d^(2*ncopies),No);
    r = zeros(No,Nh);
    for i=1:No
        for k=1:Nh
            r(i,k) = (1/(d^2))*real(trace(Ci(:,:,i)*CkU(:,:,k))); %reward function
            Xi(:,:,i) = Xi(:,:,i) + p_initial(k)*r(i,k)*Ck_tens(:,:,k);
        end
    end  

    [T{no-no_initial+1,1},score(no-no_initial+1,1),~] = testeroptimization_sdp_kcopy_seesaw(Xi,[d d],ncopies,strategy,1);
    
    
    %%%%%%%%% M3 (seesaw) %%%%%%%%% 
    T_temp = T{no-no_initial+1,1};
        
    gap = 1;
    precision = 10^(-6);
    rounds = 0;
    old_score = score(no-no_initial+1,1);
    flag_value = 0;
    
    while gap>precision
        
        rounds = rounds + 1;
                   
        %%%%% step 1 %%%%%
        
        % [estimators] = estimator_optimization(p_initial,T_temp,CkU,CkN,theta_k,ncopies);
        
        % theta_i = estimators;
        
        % Ci = zeros(d^2,d^2,No);
        % i = 0;
        % for ix = 1:no
        %     for iy = 1:no
        %         for iz = 1:no
        %             i = i + 1;
        %             thx_est = theta_i(ix,1);
        %             thy_est = theta_i(iy,2);
        %             thz_est = theta_i(iz,3);
        %             r   = sqrt(thx_est^2 + thy_est^2 + thz_est^2);
        %             % Fundamental ball: only keep r <= pi (boundary has zero measure)
        %             if r <= pi
        %                 U = expm(-1i*( thx_est*X + thy_est*Y + thz_est*Z ));
        %                 KrausU{1,1} = U;
        %                 Ci(:,:,i) = ChoiMatrix(KrausU);
        %             end
        %         end
        %     end
        % end

        % Xi = zeros(d^(2*ncopies),d^(2*ncopies),No);
        % r = zeros(No,Nh);
        % for i=1:No
        %     for k=1:Nh
        %         r(i,k) = (1/(d^2))*real(trace(Ci(:,:,i)*CkU(:,:,k))); %reward function
        %         Xi(:,:,i) = Xi(:,:,i) + p_initial(k)*r(i,k)*Tensor(CkN(:,:,k),ncopies);
        %     end
        % end  
        theta_i = estimator_update_closedform(p_initial, T_temp, Ck_tens, qk);
        %t_estimator = toc
        % Build Ci from theta_est
        Ci = zeros(d^2,d^2,No);
        for i=1:No
            th = theta_i(i,:);
            U  = expm(-1i*( th(1)*X + th(2)*Y + th(3)*Z ));
            Ci(:,:,i) = kraus2choi(U);
        end

        % Rebuild Xi (called Xk_i in your original code)
        Xi = zeros(d^(2*ncopies), d^(2*ncopies), No);
        for i=1:No
            for k=1:Nh
                r_ik = (1/(d^2))*real(trace(Ci(:,:,i)*CkU(:,:,k)));
                Xi(:,:,i) = Xi(:,:,i) + p_initial(k)*r_ik*Ck_tens(:,:,k);
            end
        end
        
        %%%%% step 2 %%%%%
        
        [T_temp,score_temp,flag] = testeroptimization_sdp_kcopy_seesaw(Xi,[d d],ncopies,strategy,1);
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
    
    T{no-no_initial+1,1} = T_temp;
    score(no-no_initial+1,1) = score_temp;        
       
end

end

function [T_i,score, solution] = testeroptimization_sdp_kcopy_seesaw(Xk_i,d,k,strategy,minmax)


din  = d(1);
dout = d(2);
No    = size(Xk_i,3);

d = repmat([din dout], 1, k);

yalmip('clear');

T_i = sdpvar((din*dout)^k,(din*dout)^k,No,'hermitian','complex');

F = [trace(sum(T_i,3))==dout^k];

score = 0;
for i=1:No
    F = F + [T_i(:,:,i)>=0];
    score = score + real(trace(T_i(:,:,i)*Xk_i(:,:,i)));
end


if strategy==1 % parallel
    F = F + [sum(T_i,3)==ProjParProcess(sum(T_i,3),d)];
elseif strategy==2 % sequential
    F = F + [sum(T_i,3)==ProjSeqProcess(sum(T_i,3),d)];
elseif strategy==3 % general
    F = F + [sum(T_i,3)==ProjGenProcess(sum(T_i,3),d)];
end

solution = optimize(F,-minmax*score,sdpsettings('solver','mosek','verbose',0,'cachesolvers',1));

T_i     = double(T_i);
score = double(score);

end

function theta_est = estimator_update_closedform(p, T, Ck_tens, qk)

No = size(T,3);
Nh = length(p);

theta_est = zeros(No,3);

for i=1:No
    lik = zeros(Nh,1);

    % Pr(i|k) = Tr[T_i * Ck^{⊗n}]
    Ti = T(:,:,i);
    for k=1:Nh
        lik(k) = real(trace(Ti * Ck_tens(:,:,k)));
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

function [X,Y,Z] = make_paulis()
%MAKE_PAULIS Generate Pauli matrices
% Returns:
%   X, Y, Z - 2x2 Pauli matrices

X = [0 1; 1 0];
Y = [0 -1i; 1i 0];
Z = [1 0; 0 -1];

end
