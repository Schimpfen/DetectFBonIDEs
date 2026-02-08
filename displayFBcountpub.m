%% plot_particle_retention_loading_flush_cycles_variability_evolution.m
% Variability-first + evolution line (ROBUST): connect MEDIANS across cycles.
% No spaghetti trajectories.
%
% Per cycle, per condition:
%   - Violin (distribution)
%   - Box (median + IQR)
%   - Raw points
%   - Median ± bootstrap 95% CI (optional)
% Across cycles:
%   - Connected median line with markers (evolution)

clear; clc; close all;

%% ---------------- User settings ----------------
CSV_FILES = { ...

};

CYCLES_TO_PLOT       = 1:5;

% Variability components
SHOW_VIOLIN          = true;
SHOW_BOX             = true;
SHOW_MEDIAN_CI       = true;
BOOT_N               = 2000;
BOX_WHISKER_MODE     = "tukey";  % "tukey" or "minmax"

% Evolution (robust)
SHOW_EVOLUTION_LINE  = true;     % connect medians across cycles
EVOL_LINE_W          = 2.0;
EVOL_MARKER_SIZE     = 4.8;

% Figure styling
FONT_NAME            = 'Helvetica';
FONT_SIZE_AX         = 8;
FONT_SIZE_LABEL      = 9;
AX_LINE_W            = 0.9;

% Colors
COL_IDES             = [0.20 0.45 0.70];
COL_MR               = [0.85 0.45 0.10];

% Raw points
POINT_SIZE           = 18;
POINT_ALPHA          = 0.65;
EDGE_ALPHA           = 0.55;
JITTER_FRAC          = 0.22;

% Violin style
VIOLIN_ALPHA         = 0.20;
VIOLIN_EDGE_ALPHA    = 0.25;
VIOLIN_MAX_WIDTH     = 0.33;

% Box style
BOX_LINE_W           = 0.9;
BOX_FACE_ALPHA       = 0.10;

% Median CI marker style
CI_LINE_W            = 1.3;
CI_CAP_W             = 0.08;

% Export
DO_EXPORT            = false;
EXPORT_BASENAME      = 'Fig_particles_retained_load_flush_cycles_variability_evolution';

% Labels
CYCLE_TERM           = 'load–flush cycle';

rng(1); % deterministic jitter

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
cond      = strings(n,1);
cycleNum  = nan(n,1);
electrode = nan(n,1);
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

    if ~isfinite(val)
        continue
    end

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

condsList = ["IDEs","MR"];
T.Condition = categorical(string(T.Condition), condsList, 'Ordinal', true);

% cycles
cyclesPresent = unique(T.Cycle(isfinite(T.Cycle)));
cycles = intersect(CYCLES_TO_PLOT(:), sort(cyclesPresent(:)))';
if isempty(cycles)
    error('No cycles found after parsing. Check filenames / CYCLES_TO_PLOT.');
end

%% === 2) Collect values per group =======================================
allVals = cell(numel(cycles), numel(condsList));
nMat    = nan(numel(cycles), numel(condsList));

for ic = 1:numel(cycles)
    cyc = cycles(ic);
    for jc = 1:numel(condsList)
        c = condsList(jc);
        vals = T.Count(T.Cycle==cyc & string(T.Condition)==c);
        vals = vals(:);
        allVals{ic,jc} = vals;
        nMat(ic,jc) = numel(vals);
    end
end

%% === 3) Plot ============================================================
pubWidth  = 3.35;
pubHeight = 2.60;

fig = figure('Units','inches','Position',[1 1 pubWidth pubHeight], ...
    'Color','w','PaperPositionMode','auto');
ax = axes('Parent',fig); hold(ax,'on'); box(ax,'off');

ax.TickDir   = 'out';
ax.LineWidth = AX_LINE_W;
ax.FontName  = FONT_NAME;
ax.FontSize  = FONT_SIZE_AX;
grid(ax,'off');

x = 1:numel(cycles);

% Two-condition x-offset
dx = 0.16;
xPos = [x(:)-dx, x(:)+dx];  % [IDEs, MR]
dxCond = abs(xPos(1,2)-xPos(1,1));

% y-range
allY = T.Count(isfinite(T.Count));
if isempty(allY), allY = 1; end
yMax = max(allY);
yPad = 0.06 * max(yMax, 1);

% --- Robust evolution summary (median per cycle) ---
medMat = nan(numel(cycles), numel(condsList));
for ic = 1:numel(cycles)
    for jc = 1:numel(condsList)
        v = allVals{ic,jc};
        v = v(isfinite(v));
        if ~isempty(v)
            medMat(ic,jc) = median(v);
        end
    end
end

% --- Draw per group: violin + box + points + median CI ---
for ic = 1:numel(cycles)
    for jc = 1:numel(condsList)
        vals = allVals{ic,jc};
        vals = vals(isfinite(vals));
        if isempty(vals), continue; end

        cName = condsList(jc);
        col = pickColor(cName, COL_IDES, COL_MR);
        x0  = xPos(ic,jc);

        % Violin (behind)
        if SHOW_VIOLIN && numel(vals) >= 3
            draw_violin(ax, x0, vals, col, VIOLIN_MAX_WIDTH, VIOLIN_ALPHA, VIOLIN_EDGE_ALPHA);
        end

        % Box
        if SHOW_BOX
            draw_box(ax, x0, vals, col, dxCond*0.42, BOX_LINE_W, BOX_FACE_ALPHA, BOX_WHISKER_MODE);
        end

        % Raw points
        jitter = (rand(size(vals)) - 0.5) * 2 * (JITTER_FRAC * dxCond);
        xs = x0 + jitter;
        hs = scatter(ax, xs, vals, POINT_SIZE, ...
            'Marker','o', 'MarkerEdgeColor','k', 'LineWidth',0.45, ...
            'MarkerFaceColor', col);
        try
            hs.MarkerFaceAlpha = POINT_ALPHA;
            hs.MarkerEdgeAlpha = EDGE_ALPHA;
        catch
        end
        hs.Annotation.LegendInformation.IconDisplayStyle = 'off';

        % Median ± bootstrap CI
        if SHOW_MEDIAN_CI && numel(vals) >= 3
            [med, lo, hi] = bootstrap_median_ci(vals, BOOT_N, 0.05);
            plot(ax, [x0 x0], [lo hi], '-', 'Color', col, 'LineWidth', CI_LINE_W);
            plot(ax, [x0-CI_CAP_W x0+CI_CAP_W], [lo lo], '-', 'Color', col, 'LineWidth', CI_LINE_W);
            plot(ax, [x0-CI_CAP_W x0+CI_CAP_W], [hi hi], '-', 'Color', col, 'LineWidth', CI_LINE_W);
            plot(ax, x0, med, 'o', 'MarkerSize', 4.6, 'MarkerFaceColor', col, 'MarkerEdgeColor','k', 'LineWidth',0.6);
        end
    end
end

% --- Evolution line (connect medians) ---
if SHOW_EVOLUTION_LINE
    for jc = 1:numel(condsList)
        col = pickColor(condsList(jc), COL_IDES, COL_MR);
        mu  = medMat(:,jc);
        msk = isfinite(mu);

        if nnz(msk) >= 2
            plot(ax, xPos(msk,jc), mu(msk), '-', 'Color', col, 'LineWidth', EVOL_LINE_W);
            plot(ax, xPos(msk,jc), mu(msk), 'o', ...
                'MarkerSize', EVOL_MARKER_SIZE, ...
                'MarkerFaceColor', col, 'MarkerEdgeColor','k', 'LineWidth',0.6);
        elseif nnz(msk) == 1
            plot(ax, xPos(msk,jc), mu(msk), 'o', ...
                'MarkerSize', EVOL_MARKER_SIZE, ...
                'MarkerFaceColor', col, 'MarkerEdgeColor','k', 'LineWidth',0.6);
        end
    end
end

% Axes
ax.XTick      = x;
ax.XTickLabel = string(cycles);

xlabel(ax, CYCLE_TERM, 'FontName',FONT_NAME, 'FontSize',FONT_SIZE_LABEL);
ylabel(ax, 'Particles retained (count)', 'FontName',FONT_NAME, 'FontSize',FONT_SIZE_LABEL);
title(ax, 'Optically detected particles', 'FontName',FONT_NAME, 'FontSize',FONT_SIZE_LABEL);

% Legend (proxy)
h1 = plot(ax, nan, nan, '-', 'Color', COL_IDES, 'LineWidth', EVOL_LINE_W);
h2 = plot(ax, nan, nan, '-', 'Color', COL_MR,   'LineWidth', EVOL_LINE_W);
lg = legend(ax, [h1 h2], {'IDEs','MR'}, 'Location','northoutside', ...
    'Orientation','horizontal', 'Box','off');
lg.FontName = FONT_NAME;
lg.FontSize = FONT_SIZE_AX;

ylim(ax, [0, yMax + yPad]);
ax.YAxis.Exponent = 0;

hold(ax,'off');

%% === 4) Print a robust summary =========================================
fprintf('\nRobust summary: median [IQR], n\n');
for ic = 1:numel(cycles)
    cyc = cycles(ic);
    fprintf('Cycle %d:\n', cyc);
    for jc = 1:numel(condsList)
        vals = allVals{ic,jc};
        vals = vals(isfinite(vals));
        if isempty(vals)
            fprintf('  %s: (no data)\n', condsList(jc));
            continue;
        end
        q = quantile(vals, [0.25 0.5 0.75]);
        fprintf('  %s: %.2f [%.2f, %.2f], n=%d\n', condsList(jc), q(2), q(1), q(3), numel(vals));
    end
end
fprintf('\n');

%% === 5) Export ==========================================================
if DO_EXPORT
    set(fig,'Renderer','painters');
    print(fig, [EXPORT_BASENAME '.pdf'], '-dpdf', '-r600');
    print(fig, [EXPORT_BASENAME '.png'], '-dpng', '-r600');
    fprintf('Exported: %s.pdf and %s.png\n', EXPORT_BASENAME, EXPORT_BASENAME);
end

%% ---- helpers -----------------------------------------------------------
function col = pickColor(condName, colIDEs, colMR)
if string(condName) == "IDEs"
    col = colIDEs;
else
    col = colMR;
end
end

function [condOut, cycleOut, electrodeOut] = parseFilename(fname)
f = replace(string(fname), '/', '\');

condToken = regexp(f, '(IDEs|MR)', 'match', 'once', 'ignorecase');
if isempty(condToken)
    error('Cannot parse condition (IDEs/MR) from filename: %s', fname);
end
if strcmpi(condToken, 'MR')
    condOut = "MR";
else
    condOut = "IDEs";
end

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

function draw_violin(ax, x0, vals, col, maxWidth, faceAlpha, edgeAlpha)
vals = vals(isfinite(vals));
if numel(vals) < 3
    return;
end

try
    [f, y] = ksdensity(vals, 'Function','pdf');
catch
    [f, y] = ksdensity(vals);
end

if all(~isfinite(f)) || max(f) <= 0
    return;
end

f = f / max(f);
w = maxWidth * f;

xL = x0 - w(:);
xR = x0 + w(:);
xx = [xL; flipud(xR)];
yy = [y(:); flipud(y(:))];

h = fill(ax, xx, yy, col, 'FaceAlpha', faceAlpha, 'EdgeColor', col, 'LineWidth', 0.6);
try
    h.EdgeAlpha = edgeAlpha;
catch
end
h.Annotation.LegendInformation.IconDisplayStyle = 'off';
end

function draw_box(ax, x0, vals, col, boxWidth, lineW, faceAlpha, whiskerMode)
vals = vals(isfinite(vals));
if isempty(vals)
    return;
end

q = quantile(vals, [0.25 0.5 0.75]);
q1 = q(1); med = q(2); q3 = q(3);
iqrV = q3 - q1;

switch string(whiskerMode)
    case "minmax"
        wLo = min(vals);
        wHi = max(vals);
    otherwise % tukey
        wLo = max(min(vals), q1 - 1.5*iqrV);
        wHi = min(max(vals), q3 + 1.5*iqrV);
end

x1 = x0 - boxWidth/2;
x2 = x0 + boxWidth/2;

hb = fill(ax, [x1 x2 x2 x1], [q1 q1 q3 q3], col, ...
    'FaceAlpha', faceAlpha, 'EdgeColor','k', 'LineWidth', lineW);
hb.Annotation.LegendInformation.IconDisplayStyle = 'off';

plot(ax, [x1 x2], [med med], '-', 'Color','k', 'LineWidth', 1.2);

plot(ax, [x0 x0], [wLo q1], '-', 'Color','k', 'LineWidth', lineW);
plot(ax, [x0 x0], [q3 wHi], '-', 'Color','k', 'LineWidth', lineW);

capW = boxWidth*0.45;
plot(ax, [x0-capW x0+capW], [wLo wLo], '-', 'Color','k', 'LineWidth', lineW);
plot(ax, [x0-capW x0+capW], [wHi wHi], '-', 'Color','k', 'LineWidth', lineW);
end

function [med, lo, hi] = bootstrap_median_ci(vals, B, alpha)
vals = vals(isfinite(vals));
n = numel(vals);
med = median(vals);

if n < 3
    lo = NaN; hi = NaN;
    return;
end

bootMed = nan(B,1);
for b = 1:B
    idx = randi(n, n, 1);
    bootMed(b) = median(vals(idx));
end

lo = quantile(bootMed, alpha/2);
hi = quantile(bootMed, 1 - alpha/2);
end
