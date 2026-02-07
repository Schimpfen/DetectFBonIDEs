%% plot_particle_retention_loading_flush_cycles_lines.m
% Publication-oriented visualization of nanoparticle retention across
% repeated loading–flush cycles (trajectory + mean band; no bars).
%
% Plot:
%   - Two conditions: IDEs vs MR
%   - Cycles: 1..5
%   - Thin lines: per-electrode trajectories (within condition)
%   - Thick line: mean across electrodes
%   - Shaded band: 95% CI (t-based) or SEM (toggle)
%   - Points: individual electrodes (jittered) at each cycle
%
% Input CSVs:
%   one or more summary_counts.csv files with header: filename,n_particles
%
% Filename examples:
%   IDEs\3R\E_4_E_4....png
%   MR\5R\E_2_E_2....png

clear; clc; close all;

%% ---------------- User settings ----------------
% One or more CSVs to combine (same header: filename,n_particles)
% Use relative paths or replace with your local data locations.
CSV_FILES           = { ...
    fullfile('data', 'summary_counts.csv'), ...
    fullfile('data', 'IDES_out', 'summary_counts.csv'), ...
    fullfile('data', 'MR_out', 'summary_counts.csv') ...
};

% Summary uncertainty on mean bands:
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
POINT_SIZE          = 20;
POINT_ALPHA         = 0.65;
EDGE_ALPHA          = 0.65;

% Jitter for raw points (relative to condition separation)
JITTER_FRAC         = 0.12;

% Electrode trajectories
SHOW_TRAJECTORIES   = true;
TRAJ_ALPHA          = 0.18;
TRAJ_LINE_W         = 0.9;

% Mean line + band
MEAN_LINE_W         = 2.0;
BAND_ALPHA          = 0.18;

% Export
DO_EXPORT           = false;
EXPORT_BASENAME     = 'Fig_particles_retained_load_flush_cycles_lines';

% Terminology (single source of truth)
CYCLE_TERM          = 'load–flush cycle';   % use en-dash

%% === 0) Read input ======================================================
if ischar(CSV_FILES) || isstring(CSV_FILES)
    CSV_FILES = cellstr(CSV_FILES);
end
if ~iscell(CSV_FILES) || isempty(CSV_FILES)
    error('CSV_FILES must be a non-empty cell array of file paths.');
end

allLines = strings(0,1);
for fi = 1:numel(CSV_FILES)
    csvFile = CSV_FILES{fi};
    if ~isfile(csvFile)
        error('Cannot find "%s".', csvFile);
    end
    dataStr = fileread(csvFile);

    lines = splitlines(string(dataStr));
    lines = strtrim(lines);
    lines = lines(lines ~= "");

    if isempty(lines)
        error('Empty file: %s', csvFile);
    end
    if ~startsWith(lines(1), "filename")
        error('First line must be the header "filename,n_particles" in: %s', csvFile);
    end

    if numel(lines) > 1
        allLines = [allLines; lines(2:end)]; %#ok<AGROW>
    end
end

%% === 1) Parse into a table =============================================
lines = allLines;
n = numel(lines);
cond      = strings(n,1);   % "IDEs" or "MR"
cycleNum  = nan(n,1);       % 1..5 (from \dR)
electrode = nan(n,1);       % electrode index
counts    = nan(n,1);
files     = strings(n,1);

k = 0;
for i = 1:numel(lines)
    row = lines(i);
    tok = split(row, ',');
    if numel(tok) ~= 2
        error('Malformed line: %s', row);
    end
    fname = strtrim(tok(1));
    val   = str2double(strtrim(tok(2)));

    k = k + 1;
    [condK, cycleK, electrodeK] = parseFilename(fname);
    cond(k)      = condK;
    cycleNum(k)  = cycleK;
    electrode(k) = electrodeK;
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

%% === 3) Plot (trajectory + mean band) ==================================
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

x = 1:numel(cycles);

% Two-condition x-offset (so IDEs and MR are side-by-side per cycle)
dx = 0.015;
xPos = [x(:)-dx, x(:)+dx]; % [IDEs, MR]

% --- Shaded uncertainty bands + mean lines ---
for jc = 1:numel(condsList)
    c   = condsList(jc);
    col = pickColor(c, COL_IDES, COL_MR);

    mu  = meanMat(:,jc);
    ee  = errMat(:,jc);

    % Band polygon (skip if all NaN)
    if any(isfinite(mu)) && any(isfinite(ee))
        yLo = mu - ee;
        yHi = mu + ee;

        % Some cycles may have NaNs: keep fill robust by masking finite
        msk = isfinite(xPos(:,jc)) & isfinite(yLo) & isfinite(yHi);
        xx  = xPos(msk,jc);
        yL  = yLo(msk);
        yH  = yHi(msk);

        if numel(xx) >= 2
            hBand = fill(ax, [xx; flipud(xx)], [yL; flipud(yH)], col, ...
                'FaceAlpha', BAND_ALPHA, 'EdgeColor','none');
            hBand.Annotation.LegendInformation.IconDisplayStyle = 'off';
        end
    end

    % Mean line
    hMean = plot(ax, xPos(:,jc), mu, '-', 'LineWidth', MEAN_LINE_W, 'Color', col);
    hMean.Annotation.LegendInformation.IconDisplayStyle = 'on';
end

% --- Per-electrode trajectories (thin lines) ---
if SHOW_TRAJECTORIES
    for jc = 1:numel(condsList)
        c   = condsList(jc);
        col = pickColor(c, COL_IDES, COL_MR);

        elecs = unique(T.Electrode(string(T.Condition)==c));
        for e = elecs(:)'
            idx = (string(T.Condition)==c) & (T.Electrode==e);
            cyc = T.Cycle(idx);
            yy  = T.Count(idx);

            [cycS, order] = sort(cyc);
            yyS = yy(order);

            % Map cycle numbers to x positions for this condition
            xLine = xPos(cycS, jc);

            htr = plot(ax, xLine, yyS, '-', 'LineWidth', TRAJ_LINE_W, 'Color', col);
            try
                htr.Color(4) = TRAJ_ALPHA; % RGBA alpha (newer MATLAB)
            catch
                % older MATLAB: ignore alpha
            end
            htr.Annotation.LegendInformation.IconDisplayStyle = 'off';
        end
    end
end

% --- Raw points (jittered) ---
rng(1); % deterministic jitter
for ic = 1:numel(cycles)
    for jc = 1:numel(condsList)
        vals = allVals{ic,jc};
        if isempty(vals), continue; end

        col = pickColor(condsList(jc), COL_IDES, COL_MR);

        % jitter amplitude based on the separation between conditions
        dxCond = abs(xPos(ic,2) - xPos(ic,1));
        jitter = (rand(size(vals)) - 0.5) * 2 * (JITTER_FRAC * dxCond);

        xs = xPos(ic,jc) + jitter;

        hs = scatter(ax, xs, vals, POINT_SIZE, ...
            'Marker','o', 'MarkerEdgeColor','k', 'LineWidth',0.5, ...
            'MarkerFaceColor', col);

        try
            hs.MarkerFaceAlpha = POINT_ALPHA;
            hs.MarkerEdgeAlpha = EDGE_ALPHA;
        catch
        end
        hs.Annotation.LegendInformation.IconDisplayStyle = 'off';
    end
end

% Axes labels & ticks
ax.XTick      = x;
ax.XTickLabel = string(cycles);

xlabel(ax, sprintf('%s', CYCLE_TERM), 'FontName',FONT_NAME, 'FontSize',FONT_SIZE_LABEL);
ylabel(ax, 'Particles retained (count)',      'FontName',FONT_NAME, 'FontSize',FONT_SIZE_LABEL);
title('Optically detected particles',      'FontName',FONT_NAME, 'FontSize',FONT_SIZE_LABEL);

% Legend: show mean lines only (one per condition)
lg = legend(ax, {'IDEs','MR'}, 'Location','northoutside', ...
    'Orientation','horizontal', 'Box','off');
lg.FontName = FONT_NAME;
lg.FontSize = FONT_SIZE_AX;

% Y limits
yMax = max([meanMat(:)+errMat(:); T.Count], [], 'omitnan');
if ~isfinite(yMax), yMax = 1; end
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
% loading–flush cycles. Thin lines show per-electrode trajectories; points
% show individual observations. Thick lines show the mean per cycle with a
% shaded band indicating mean ± 95% CI across electrodes.

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

function [condOut, cycleOut, electrodeOut] = parseFilename(fname)
% Robustly parse condition (IDEs/MR), cycle (1..), and electrode (E#)
f = replace(string(fname), '/', '\');

% Condition: IDEs or MR anywhere in path/filename
condToken = regexp(f, '(IDEs|MR)', 'match', 'once', 'ignorecase');
if isempty(condToken)
    error('Cannot parse condition (IDEs/MR) from filename: %s', fname);
end
if strcmpi(condToken, 'MR')
    condOut = "MR";
else
    condOut = "IDEs";
end

% Cycle: prefer folder like "\1R\"; fallback to "_R1_"
m = regexp(f, '(^|[\\/])(\d+)R([\\/])', 'tokens', 'once');
if ~isempty(m)
    cycleOut = str2double(m{2});
else
    m = regexp(f, '[\\/_]R(\d+)[\\/_]', 'tokens', 'once');
    if isempty(m)
        m = regexp(f, 'R(\d+)', 'tokens', 'once');
    end
    if isempty(m)
        error('Cannot parse cycle from filename: %s', fname);
    end
    cycleOut = str2double(m{1});
end

% Electrode: prefer "E_#_E_#"; fallback to "E_#" or "E#"
m = regexp(f, 'E_(\d+)_E_\1', 'tokens', 'once');
if isempty(m)
    m = regexp(f, 'E_(\d+)', 'tokens', 'once');
end
if isempty(m)
    m = regexp(f, 'E(\d+)', 'tokens', 'once');
end
if isempty(m)
    error('Cannot parse electrode from filename: %s', fname);
end
electrodeOut = str2double(m{1});
end
