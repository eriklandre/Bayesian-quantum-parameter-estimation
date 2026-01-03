function [T_i,score, solution] = testeroptimization_sdp_kcopy_seesaw(Xk_i,d,k,strategy,minmax)
% Xk_i: Choi matrix of the k-copy channel to be optimized
% d: [din dout] dimensions of the input and output systems
% k: number of copies
% strategy: 1 for parallel, 2 for sequential, 3 for general
% minmax: 1 for maximization, -1 for minimization

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