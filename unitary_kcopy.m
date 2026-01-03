function [T,score] = unitary_kcopy(ncopies, method, strategy)
% ncopies: number of copies of the channel
% set method = 1, 2, or 3, to choose desired method
% set strategy = 1, 2, or 3, to choose desired strategy (PARALLEL, SEQUENTIAL, GENERAL)

d = 2;

[X,Y,Z] = make_paulis;

no_initial = 2;
no_final   = 2;

score = zeros(no_final-no_initial+1,1);
T     = cell(no_final-no_initial+1,1);

for no=no_initial:no_final

    no = no
    
    %%%%%%%%% M1 %%%%%%%%%
    if method==1
        nh = no;
        
    %%%%%%%%% M2 and M3 %%%%%%%%%   
    else
        nh = 10; 
    end
    
    No = no^3;
    Nh = nh^3;
   
    estimators = -pi:2*pi/(no):pi;      % uniform distribution of the estimator 
    discretization = -pi:2*pi/(nh):pi;  % uniform discretization of the true parameter 
    
    theta_i = zeros(no,3);
    theta_k = zeros(nh,3);
    
    theta_i(:,1) = estimators(1:no); % theta_i x
    theta_i(:,2) = estimators(1:no); % theta_i y
    theta_i(:,3) = estimators(1:no); % theta_i z
    
    theta_k(:,1) = discretization(1:nh); % theta_k x
    theta_k(:,2) = discretization(1:nh); % theta_k y
    theta_k(:,3) = discretization(1:nh); % theta_k z
    
    p(1:Nh,1) = 1/Nh;       % uniform prior
    
%     % gaussian prior %
%     pk = normpdf(theta_k,0,1);
%     pk = pk./sum(pk);
% %     
%     p = zeros(Nh,1);
%     k=0;
%     for xk=1:nh
%         for yk=1:nh
%             for zk=1:nh
%                 k = k+1;
%                 p(k,1) = pk(xk)*pk(yk)*pk(zk);
%             end
%         end
%     end
    Ck = zeros(d^2,d^2,Nh);
    k=0;
    for kx=1:nh
        for ky=1:nh
            for kz=1:nh
                k = k+1;
                Ck(:,:,k) = kraus2choi(expm(1i*(theta_k(kx,1)*X+theta_k(ky,2)*Y+theta_k(kz,3)*Z)));     % Choi matrices of the true channels
            end 
        end
    end
    
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

    Xk_i = zeros(d^(2*ncopies),d^(2*ncopies),No);
    r = zeros(No,Nh);
    for i=1:No
        for k=1:Nh
            r(i,k) = (1/(d^2))*real(trace(Ci(:,:,i)*Ck(:,:,k)));        % reward function
            Xk_i(:,:,i) = Xk_i(:,:,i) + p(k)*r(i,k)*Tensor(Ck(:,:,k),ncopies);      % computing the Xi's
        end
    end    

    [T{no-no_initial+1,1},score(no-no_initial+1,1),~] = testeroptimization_sdp_kcopy_seesaw(Xk_i,[d d],ncopies,strategy,1);     % SDP optimization to get the optimal tester and score
    
    
    %%%%%%%%% M3: SEESAW algorithm (takes as input the optimal tester and optimizes over estimators) %%%%%%%%% 
    if method==3    
        
        T_temp = T{no-no_initial+1,1};
        
        gap = 1;
        precision = 10^(-6);
        rounds = 0;
        old_score = score(no-no_initial+1,1);
        flag_value = 0;
        
        while gap>precision
            
            rounds = rounds + 1;
            
            [estimators,~] = estimator_optimization(p,T_temp,Ck,theta_i,ncopies);       % optimization over the estimator
            
            theta_i = estimators;
            
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
            
            Xk_i = zeros(d^(2*ncopies),d^(2*ncopies),No);
            r = zeros(No,Nh);
            for i=1:No
                for k=1:Nh
                    r(i,k) = (1/(d^2))*real(trace(Ci(:,:,i)*Ck(:,:,k))); %reward function
                    Xk_i(:,:,i) = Xk_i(:,:,i) + p(k)*r(i,k)*Tensor(Ck(:,:,k),ncopies);
                end
            end
            
            [T_temp,score_temp,flag] = testeroptimization_sdp_kcopy_seesaw(Xk_i,[d d],ncopies,strategy,1);      % SDP optimization to get the optimal tester and score
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

end

function [estimators,score] = estimator_optimization(p,T,Ck,theta_i,ncopies)

options = optimset('MaxFunEvals',50,'MaxIter',50,'TolX',1e-5,'TolFun',1e-5);

x0 = theta_i;

[estimators,fval] = fminsearch(@(x)objectivefcn(x,p,T,Ck,ncopies),x0,options);

score = -fval;

end

function score = objectivefcn(theta_i,p,T,Ck,ncopies)

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
        score = score + p(k)*r(i,k)*real(trace(T(:,:,i)*Tensor(Ck(:,:,k),ncopies)));
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



