fileNames = {
    'thermometry_sequential.mat', ...
    'thermometry_greedy.mat', ...
    'thermometry_nonadaptive.mat',
};
% Define marker styles and labels
markers = {'-o', '-o', '-x'}; %'-s', '-x'
labels = {
    'Sequential', ...
    'Greedy adaptive', ...
    'Non-adaptive',
};

colors = {
    "#DD5400",...
    "#3BAA32", ...
    "#1F77B4", ...
};

% Create figure
figure;
hold on;

% Loop through files and plot
for i = 1:3
    % Load the file
    data = load(fileNames{i});
    
    % Assume variables are named like this inside the files:
    % data.x  → x-axis values (e.g., NO^(1/3))
    % data.y  → y-axis values (e.g., scores)
    
    % If not, adjust accordingly below:
    x = data.t_vals;
    y = data.scores;
    
    % Plot
% Renormalize t_vals (x) from [0,100] to [0,2]
if ~isempty(x)
    xmin = min(x(:));
    xmax = max(x(:));
    if xmax > xmin
        % map x in [xmin,xmax] -> [0,2], assuming xmin corresponds to 0 and xmax to 100
        % but enforce original range expected [0,100]
        % compute normalized in [0,1] then scale to [0,2]
        x = (x - xmin) ./ (xmax - xmin) * 2;
    end
end
    plot(x, y, markers{i}, 'Color', colors{i},'LineWidth', 2, 'MarkerSize', 8);
end
xlabel('$J (\epsilon) t$', 'Interpreter', 'latex', 'FontSize', 24, 'Color', 'k');
ylabel('$\tilde{\mathcal{S}}$ (minimization)', 'Interpreter', 'latex', 'FontSize', 24, 'Color', 'k');

% Legend
%legend(labels, 'Interpreter', 'latex', 'FontSize', 24, 'Location', 'best', 'Box', 'Off');

% Axes settings
%ylim([0.45 0.65]); % adjust if needed
ax = gca;
ax.FontSize = 24;
ax.TickLabelInterpreter = 'latex';
ax.XColor = 'k';
ax.YColor = 'k';

grid on;
hold off;
box on;

%exportgraphics(gcf, 'greedyvssequential_time_nonadaptive.pdf', 'ContentType', 'vector');
% savefig('thermometry_new.fig')