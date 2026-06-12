tic
method = 3;
ncopies = 2;
strategy = 1;
no = 20;
p_vals = linspace(0,1,21);
scores = zeros(length(p_vals), 1);
%scores_adaptive = zeros(length(No_values), 1);
for i = 1:length(p_vals)
    p = p_vals(i);
    [~,score] = phaseestimation_and_noise_kcopy(no, no, ncopies, strategy, p);
    scores(i) = score;
end
disp(scores);
toc
