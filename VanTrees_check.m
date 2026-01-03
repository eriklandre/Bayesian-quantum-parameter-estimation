function diag = VanTrees_check(theta_k, Ck, T, ncopies, est_angles, p_prior)

    theta_k = theta_k(:).';             % true parameter distribution
    p_prior = p_prior(:).';             % prior probability distribution
    p_prior = p_prior / sum(p_prior);   % normalization condition
    No      = size(T,3);                % number of outputs of the measurement
    Nh      = numel(theta_k);           % number of hypothesis 

    Q = zeros(No,Nh);                   
    for k = 1:Nh
        W = Tensor(Ck(:,:,k), ncopies); 
        for i = 1:No
            Q(i,k) = real(trace(T(:,:,i) * W)); % outcome likelihoods Q(i,k)
        end
    end
    Q = Q ./ sum(Q,1);                  % normalization condition

    % prior-averaged FI

    dth   = theta_k(2) - theta_k(1);   % true parameter discretization
    dQ    = zeros(size(Q));            % outcome likelihood discretization

    if Nh >= 2
        % boundary conditions
        dQ(:,1) = (Q(:,2) - Q(:,1)) / dth;
        dQ(:,Nh) = (Q(:,Nh) - Q(:,Nh-1)) / dth;
    end

    for k = 2:Nh-1
        % interior discretization
        dQ(:,k) = (Q(:,k+1) - Q(:,k-1)) / (2*dth);
    end

    Jtheta   = sum( (dQ.^2) ./ Q, 1 );   % FI
    EJ       = sum(p_prior .* Jtheta);   % prior-averaged FI

    % prior FI

    pp = p_prior(:);
    dp = zeros(Nh,1);

    if Nh >= 2
        % boundary conditions
        dp(1) = (pp(2) - pp(1)) / dth;
        dp(Nh) = (pp(Nh) - pp(Nh-1)) / dth;
    end

    for k = 2:Nh-1
        % interior discretization
        dp(k) = (pp(k+1) - pp(k-1)) / (2*dth);
    end

    F0 = sum( (dp.^2) ./ pp );          % prior FI

    % EMSD
    est_angles = est_angles(:);
    mse_theta   = zeros(Nh,1);                 
    for k = 1:Nh
        d = est_angles - theta_k(k);          
        mse_theta(k) = sum( Q(:,k) .* (d.^2) ); % MSE
    end
    EMSD_scalar = sum( p_prior(:) .* mse_theta(:) );    % EMSD

    % Van Trees bound 
    vanTreesLB = 1 / (F0 + EJ);

    % pack results
    format long
    diag = struct();
    diag.F0         = F0;
    diag.EJ         = EJ;
    diag.vanTreesLB = vanTreesLB;
    diag.EMSD       = EMSD_scalar;
    diag.summary_str = sprintf('F0 = %.3e | E[F] = %.3e | VT-LB = %.3e | EMSD = %.3e', ...
                               F0, EJ, vanTreesLB, EMSD_scalar);
    disp(diag.F0);

    fprintf('%s\n', diag.summary_str);
end
