labels = {
    'PAR = SEQ = ICO', ...
    'Greedy adaptive', ...
    'Non-adaptive',
};

scores_adaptive = [0.5000, 0.6229, 0.7044, 0.7712, 0.8144, 0.8457]; % obtained from unitary_and_noise_greedy.m for 10k MC rounds
scores_nonadaptive = [0.5000, 0.5666, 0.6246, 0.6686, 0.7006, 0.7285]; % obtained from unitary_and_noise_nonadaptive.m for 10k MC rounds

copies = linspace(1,6,6);
copies_nonadaptive = linspace(1,6,6);
disp(analytical_scores(copies));

% Create figure
figure;
hold on;
box on;

% Axis labels
plot(copies_nonadaptive, analytical_scores(copies_nonadaptive), '--s', 'Color', 'k', 'LineWidth', 3, 'MarkerSize', 12);
plot(copies, scores_adaptive, '-o', 'Color', '#3BAA32', 'LineWidth', 3, 'MarkerSize', 12);
plot(copies_nonadaptive, scores_nonadaptive, '-x', 'Color', '#0072BD', 'LineWidth', 3, 'MarkerSize', 12);
xlabel('$k$', 'Interpreter', 'latex', 'FontSize', 24, 'Color', 'k');
ylabel('$\tilde{\mathcal{S}}$ (maximization)', 'Interpreter', 'latex', 'FontSize', 24, 'Color', 'k');

% Legend
legend(labels, 'Interpreter', 'latex', 'FontSize', 21, 'Location', 'best', 'Box', 'Off');

% Axes settings
%ylim([0.45 0.75]); % adjust if needed
ax = gca;
ax.FontSize = 24;
ax.TickLabelInterpreter = 'latex';
ax.XColor = 'k';
ax.YColor = 'k';

grid on;
hold off;

%exportgraphics(gcf, 'su2_greedy_kcopies.pdf', 'ContentType', 'vector');

function [scores] = analytical_scores(k_copies)
    % Analytical scores for different k_copies
    scores = zeros(length(k_copies), 1);
    for i = 1:length(k_copies)
        k = k_copies(i);
        scores(i) = cos(pi/(k+3))^2; % Example formula
    end
end

% savefig('kcopies_new.fig')
