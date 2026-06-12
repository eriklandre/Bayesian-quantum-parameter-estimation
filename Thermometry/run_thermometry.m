tic
method = 3;
no_initial = 20;
no_final = 20;
No = 4;
ncopies = 2;
strategy = 2;
t_vals = linspace(1, 100, 100);                            % collect every 10th time index
score_t = zeros(no_final-no_initial+1, numel(t_vals));
for j = 1:numel(t_vals)
    t = t_vals(j);
    %[~, scores] = thermometry_kcopy(method, t, no_initial, no_final, ncopies, strategy);
    score_adaptive = thermometry_greedy(No, ncopies, 1000000, t);
    %score_nonadaptive = thermometry_nonadaptive(No, ncopies, 1000000, t);
    score_t(:, j) = score_adaptive; 
    %score_t(:, j) = scores(:, t_vals(j));
    % score_t(:, j) = score_nonadaptive;
    disp(score_adaptive);
end

disp(score_t);

%save('thermometry_sequential.mat', 't_values', 'score_t'); %this filename is for fully quantum strategies
%save('thermometry_greedy.mat', 't_values', 'score_t'); %this filename is for greedy strategies
%save('thermometry_nonadaptive.mat', 't_values', 'score_t'); %this filename is for non-adaptive strategies
toc
