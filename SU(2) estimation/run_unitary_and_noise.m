profile clear
profile on
tic
method = 3;
ncopies = 2;
strategy = 3;
n_monte_carlo = 10000;
p_values = linspace(0,1,21);
score = zeros(length(p_values),1);
for idx = 1:length(p_values)
    p = p_values(idx);
    [~,score(idx)] = unitary_and_noise_kcopy(method, ncopies, strategy, p);
    %score(idx) = unitary_and_noise_greedy(10, ncopies, n_monte_carlo, p);
    %score(idx) = unitary_and_noise_nonadaptive(10, ncopies, n_monte_carlo, p);
end
disp(score);
toc
profile off
profile viewer
