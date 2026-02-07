%% plot_particle_retention_loading_flush_cycles.m
% Publication-oriented visualization of nanoparticle retention across
% repeated loading–flush cycles.
%
% Concept:
%   One "loading–flush cycle" = particle loading step followed by a flush
%   intended to remove non-retained particles.
%
% Plot:
%   - Two conditions: IDEs vs MR
%   - Cycles: 1..5
%   - Bars: mean across electrodes
%   - Error bars: 95% CI (t-based) or SEM (toggle)
%   - Points: individual electrodes (jittered)
%   - Optional: per-electrode trajectories (spaghetti lines) within condition
%
% Input CSV:
%   summary_counts.csv with header: filename,n_particles
%
% Filename examples:
%   IDEs\3R\E_4_E_4....png
%   MR\5R\E_2_E_2....png

clear; clc; close all;

%% ---------------- User settings ----------------
CSV_FILE            = 'summary_counts.csv';

% Summary uncertainty on bars:
USE_95CI            = true;     % true: t-based 95% CI; false: SEM
CI_ALPHA            = 0.05;

% Figure styling (compact, journal-friendly defaults)
FONT_NAME           = 'Helvetica';
FONT_SIZE_AX        = 8;
FONT_SIZE_LABEL     = 9;
AX_LINE_W           = 0.9;

% Colors (muted, high contrast)
COL_IDES            = [0.20 0.45 0.70];
COL_MR              = [0.85 0.45 0.10];

% Raw points
POINT_SIZE          = 22;
POINT_ALPHA         = 0.75;
EDGE_ALPHA          = 0.85;

% Jitter for raw points (relative to condition separation)
JITTER_FRAC         = 0.12;

% Optional electrode trajectories across cycles
SHOW_TRAJECTORIES   = true;
TRAJ_ALPHA          = 0.18;
TRAJ_LINE_W         = 0.8;

% Export
DO_EXPORT           = true;
EXPORT_BASENAME     = 'Fig_particles_retained_loading_flush_cycles';

% Terminology (single source of truth)
CYCLE_TERM          = 'loading–flush cycle';   % use en-dash

%% === 0) Read input ======================================================
if ~isfile(CSV_FILE)
    error('Cannot find "%s".', CSV_FILE);
end
DATA_STR = fileread(CSV_FILE);

%% === 1) Parse into a table =============================================
lines = splitlines(string(DATA_STR));
lines = strtrim(lines);
lines = lines(lines ~= "");

if ~startsWith(lines(1), "filename")
    error('First line must be the header "filename,n_particles".');
end

n = numel(lines) - 1;
cond      = strings(n,1);   % "IDEs" or "MR"
cycleNum  = nan(n,1);       % 1..5 (from \dR)
electrode = nan(n,1);       % 1..6
counts    = nan(n,1);
files     = strings(n,1);

k = 0;
for i = 2:numel(lines)
    row = lines(i);
    tok = split(row, ',');
    if numel(tok) ~= 2
        error('Malformed line: %s', row);
    end
    fname = strtrim(tok(1));
    val   = str2double(strtrim(tok(2)));

    % Strict pattern first (enforces E_x_E_x)
    m = regexp(fname, '^(IDEs|MR)\\(\d)R\\E_(\d+)_E_\3\..*$', 'tokens', 'once');
    if isempty(m)
        % Looser fallback: grab condition, cycle, first electrode occurrence
        m = regexp(fname, '^(IDEs|MR)\\(\d)R\\E_(\d+)_', 'tokens', 'once');
    end
    if isempty(m)
        error('Cannot parse condition/cycle/electrode from filename: %s', fname);
    end

    k = k + 1;
    cond(k)      = string(m{1});
    cycleNum(k)  = str2double(m{2});
    electrode(k) = str2double(m{3});
    counts(k)    = val;
    files(k)     = fname;
end

T = table(cond(1:k), cycleNum(1:k), electrode(1:k), counts(1:k), files(1:k), ...
    'VariableNames', {'Condition','Cycle','Electrode','Count','Filename'});

% Consistent ordering for plotting
condsList = ["IDEs","MR"];
T.Condition = categorical(string(T.Condition), condsList, 'Ordinal', true);

cycles = (1:5)';

%% === 2) Aggregate: mean + CI/SEM =======================================
meanMat = nan(numel(cycles), numel(condsList));
errMat  = nan(numel(cycles), numel(condsList));   % SEM or CI half-width
nMat    = nan(numel(cycles), numel(condsList));
allVals = cell(numel(cycles), numel(condsList));

for ic = 1:numel(cycles)
    cyc = cycles(ic);
    for jc = 1:numel(condsList)
        c = condsList(jc);
        vals = T.Count(T.Cycle==cyc & string(T.Condition)==c);
        vals = vals(:);

        allVals{ic,jc} = vals;
        nMat(ic,jc)    = numel(vals);
        meanMat(ic,jc) = mean(vals, 'omitnan');

        if numel(vals) >= 2
            s  = std(vals, 'omitnan');
            se = s / sqrt(numel(vals));
            if USE_95CI
                tcrit = tinv(1 - CI_ALPHA/2, numel(vals)-1);
                errMat(ic,jc) = tcrit * se;
            else
                errMat(ic,jc) = se;
            end
        else
            errMat(ic,jc) = NaN;
        end
    end
end

%% === 3) Plot ============================================================
pubWidth  = 3.35;  % inches (single column-ish)
pubHeight = 2.35;

fig = figure('Units','inches','Position',[1 1 pubWidth pubHeight], ...
    'Color','w','PaperPositionMode','auto');
ax = axes('Parent',fig); hold(ax,'on');
box(ax,'off');

% Axes style
ax.TickDir   = 'out';
ax.LineWidth = AX_LINE_W;
ax.FontName  = FONT_NAME;
ax.FontSize  = FONT_SIZE_AX;
grid(ax,'off');

% Bars (mean)
x = 1:numel(cycles);
barW = 0.70;

b = bar(ax, x, meanMat, barW, 'EdgeColor','k', 'LineWidth',0.7);
b(1).FaceColor = COL_IDES;
b(2).FaceColor = COL_MR;
b(1).FaceAlpha = 0.25;
b(2).FaceAlpha = 0.25;

drawnow; % enables XEndPoints

% Error bars (CI or SEM)
for jc = 1:numel(condsList)
    xe = b(jc).XEndPoints;
    he = errorbar(ax, xe, meanMat(:,jc), errMat(:,jc), ...
        'k', 'LineStyle','none', 'LineWidth',0.9, 'CapSize',6);
    he.Annotation.LegendInformation.IconDisplayStyle = 'off';
end

% Optional: per-electrode trajectories across cycles within each condition
if SHOW_TRAJECTORIES
    for jc = 1:numel(condsList)
        c = condsList(jc);
        col = pickColor(c, COL_IDES, COL_MR);

        elecs = unique(T.Electrode(string(T.Condition)==c));
        for e = elecs(:)'
            idx = (string(T.Condition)==c) & (T.Electrode==e);
            cyc = T.Cycle(idx);
            yy  = T.Count(idx);

            [cycS, order] = sort(cyc);
            yyS = yy(order);

            xLine = b(jc).XEndPoints(cycS);
            htr = plot(ax, xLine, yyS, '-', 'LineWidth',TRAJ_LINE_W, 'Color',col);
            try
                htr.Color(4) = TRAJ_ALPHA; % RGBA alpha (newer MATLAB)
            catch
                % older MATLAB: ignore alpha
            end
            htr.Annotation.LegendInformation.IconDisplayStyle = 'off';
        end
    end
end

% Raw electrode points (jittered)
rng(1); % deterministic jitter
for ic = 1:numel(cycles)
    for jc = 1:numel(condsList)
        vals = allVals{ic,jc};
        if isempty(vals), continue; end

        xC  = b(jc).XEndPoints(ic);
        col = pickColor(condsList(jc), COL_IDES, COL_MR);

        % jitter amplitude based on condition separation at this cycle
        dxCond = abs(b(2).XEndPoints(ic) - b(1).XEndPoints(ic));
        jitter = (rand(size(vals)) - 0.5) * 2 * (JITTER_FRAC * dxCond);
        xs = xC + jitter;

        hs = scatter(ax, xs, vals, POINT_SIZE, ...
            'Marker','o', 'MarkerEdgeColor','k', 'LineWidth',0.5, ...
            'MarkerFaceColor',col);

        try
            hs.MarkerFaceAlpha = POINT_ALPHA;
            hs.MarkerEdgeAlpha = EDGE_ALPHA;
        catch
        end

        hs.Annotation.LegendInformation.IconDisplayStyle = 'off';
    end
end

% X axis: show cycles as plain integers (no coded suffix)
ax.XTick      = x;
ax.XTickLabel = string(cycles);

xlabel(ax, sprintf('%s number', CYCLE_TERM), 'FontName',FONT_NAME, 'FontSize',FONT_SIZE_LABEL);
ylabel(ax, 'Particles retained (count)',      'FontName',FONT_NAME, 'FontSize',FONT_SIZE_LABEL);

% Legend (minimal)
lg = legend(ax, {'IDEs','MR'}, 'Location','northoutside', ...
    'Orientation','horizontal', 'Box','off');
lg.FontName = FONT_NAME;
lg.FontSize = FONT_SIZE_AX;

% Y limits
yMax = max([meanMat(:)+errMat(:); T.Count], [], 'omitnan');
ylim(ax, [0, 1.08*yMax]);
ax.YAxis.Exponent = 0;

hold(ax,'off');

%% === 4) Print a stats summary ==========================================
uncTerm = ternary(USE_95CI, '95% CI', 'SEM');

fprintf('\nSummary (mean ± %s) particles retained per %s:\n', uncTerm, CYCLE_TERM);
for ic = 1:numel(cycles)
    fprintf('  Cycle %d  IDEs: %.2f ± %.2f (n=%d)  |  MR: %.2f ± %.2f (n=%d)\n', ...
        cycles(ic), meanMat(ic,1), errMat(ic,1), nMat(ic,1), ...
        meanMat(ic,2), errMat(ic,2), nMat(ic,2));
end
fprintf('\n');

%% === 5) Export ==========================================================
if DO_EXPORT
    set(fig,'Renderer','painters'); % vector output
    print(fig, [EXPORT_BASENAME '.pdf'], '-dpdf', '-r600');
    print(fig, [EXPORT_BASENAME '.png'], '-dpng', '-r600');
    fprintf('Exported: %s.pdf and %s.png\n', EXPORT_BASENAME, EXPORT_BASENAME);
end

%% === Caption text suggestion (paste into manuscript) =====================
% Particles retained on IDEs and microrails (MR) after successive
% loading–flush cycles. Each cycle comprises nanoparticle loading followed by
% a flush intended to remove non-retained particles. Bars show mean ± 95% CI
% across electrodes (n = 6); points indicate individual electrodes.

%% ---- helpers -----------------------------------------------------------
function out = ternary(cond, a, b)
if cond, out = a; else, out = b; end
end

function col = pickColor(condName, colIDEs, colMR)
if string(condName) == "IDEs"
    col = colIDEs;
else
    col = colMR;
end
end
