% =========================================================================
%  PARTIE 1.3 - Filtre de Wiener, hyperplan separateur, meilleur separateur,
%  RSB et TEB.            (Evaluation TP "Classification et Filtrage Bayesien")
%
%  Fichiers necessaires (meme dossier) :
%    centres_classes.m, generer_observations.m, filtre_wiener.m,
%    classifieur_bayes.m, hyperplan_max_marge.m, hyperplan_min_teb.m,
%    teb_lineaire_theorique.m, tracer_classes_2D.m, tracer_classes_3D.m
%
%  Modele :  x(n) = CC*D(n) + v(n),  d(n) iid dans {+1,-1},  v ~ N(0,sigma2)
%  RSB    :  10*log10( c'*Rd*c / sigma2 ),  avec Rd = I  ->  ||c||^2 / sigma2
%
%  Les figures sont sauvegardees dans ./figures (reprises dans le rapport).
% =========================================================================
clear; clc; close all;
rng(42);
if ~exist('figures', 'dir'), mkdir('figures'); end
sauver  = @(nom) print(gcf, fullfile('figures', nom), '-dpng', '-r150');
teb_sim = @(w, X, d) mean((2*((w(:).' * X) > 0) - 1) ~= d);   % decision sign(w'x)
marge   = @(w, Xp) min(w(:).' * Xp) / norm(w);                % marge geometrique

%% ---------------------------------------------------------------- 0. Donnees
c = [1; 0.5];       % canal de reference (M = 2)
M = length(c);
P = 2;              % dimension du regresseur
N = 1e4;            % nombre d'observations (validation)
variances = [0.01 0.2 1.0];

[CC, X_plus, X_minus] = centres_classes(c, P);
fprintf('Matrice de convolution CC :\n'); disp(CC);
fprintf('Centres de E+ :\n'); disp(X_plus);

%% ----------------------------------------------- 1. Coefficients de Wiener
fprintf('\n===== 1. FILTRE DE WIENER  (w = (CC*CC'' + sigma2*I)^-1 * CC(:,1)) =====\n');
W = zeros(P, numel(variances));
for q = 1:numel(variances)
    W(:, q) = filtre_wiener(CC, variances(q));
    fprintf('sigma2 = %-5.2f  ->  w = [%8.4f %8.4f],  angle de la normale = %6.2f deg\n', ...
        variances(q), W(1, q), W(2, q), atan2d(W(2, q), W(1, q)));
end
fprintf('Limite sigma2 -> inf : w ~ CC(:,1) = [c(0); 0] (filtre adapte)\n');

% Verification par estimation empirique : w_hat = R_hat^-1 p_hat
s2 = 0.2;
[Xv, dv] = generer_observations(c, P, N, s2);
w_emp = (Xv * Xv.' / N) \ (Xv * dv.' / N);
fprintf('Verification (sigma2=0.2) : theorique [%.4f %.4f]  empirique [%.4f %.4f]\n', ...
    W(1, 2), W(2, 2), w_emp(1), w_emp(2));

%% ---------- 2. Hyperplan de Wiener vs frontiere de Bayes pour 3 variances
figure(1); clf; set(gcf, 'Position', [50 50 1400 450]);
for q = 1:numel(variances)
    subplot(1, 3, q);
    tracer_classes_2D(c, variances(q), 2000, 3, q == 1);
end
sauver('fig1_wiener_bayes_sigma.png');

%% ---------------------------- 3. Meilleur separateur et degre de separabilite
fprintf('\n===== 3. HYPERPLANS SEPARATEURS (c = [%s]) =====\n', num2str(c.', '%g '));
[w_mm, delta, lambda] = hyperplan_max_marge(X_plus);
w_mm_n = w_mm / delta;
idx_sv = find(lambda > 1e-6);
fprintf('Max-marge : delta = dist(0, conv(E+)) = %.4f, direction w = [%.4f %.4f]\n', ...
        delta, w_mm_n(1), w_mm_n(2));
fprintf('Point de conv(E+) le plus proche de 0 : x* = [%.4f %.4f]\n', w_mm(1), w_mm(2));
fprintf('Centres "de support" (lambda>0) : indices %s\n', mat2str(idx_sv.'));
fprintf('Separabilite lineaire : %s\n', mat2str(delta > 1e-3));

fprintf('\n sigma2 | rho=delta/sigma | marge Wiener | marge MaxMarge | TEBth Wiener | TEBth MaxMarge | TEBth MinTEB\n');
for q = 1:numel(variances)
    s2 = variances(q);
    w_mt = hyperplan_min_teb(X_plus, s2, [W(:, q), w_mm_n]);
    fprintf(' %5.2f  |  %9.3f      |  %9.4f   |  %9.4f     |  %.3e   |  %.3e     |  %.3e\n', ...
        s2, delta/sqrt(s2), marge(W(:, q), X_plus), marge(w_mm_n, X_plus), ...
        teb_lineaire_theorique(W(:, q), X_plus, s2), ...
        teb_lineaire_theorique(w_mm_n, X_plus, s2), ...
        teb_lineaire_theorique(w_mt, X_plus, s2));
end

% --- Figure 2 : geometrie (enveloppe convexe, point le plus proche, marge)
s2 = 0.2;  wW = W(:, 2) / norm(W(:, 2));
figure(2); clf; hold on; grid on; axis equal;
h = convhull(X_plus(1, :).', X_plus(2, :).');
g1 = fill( X_plus(1, h),  X_plus(2, h), 'b', 'FaceAlpha', 0.12, 'EdgeColor', 'b');
g2 = fill(-X_plus(1, h), -X_plus(2, h), 'r', 'FaceAlpha', 0.12, 'EdgeColor', 'r');  % conv(E-) = -conv(E+)
g3 = plot(X_plus(1, :),  X_plus(2, :),  'b+', 'MarkerSize', 12, 'LineWidth', 2);
g4 = plot(X_minus(1, :), X_minus(2, :), 'ro', 'MarkerSize', 9,  'LineWidth', 2);
g5 = plot(X_plus(1, idx_sv), X_plus(2, idx_sv), 'ks', 'MarkerSize', 14, 'LineWidth', 1.5);
g6 = plot(w_mm(1), w_mm(2), 'kp', 'MarkerSize', 14, 'MarkerFaceColor', 'y');
plot([0 w_mm(1)], [0 w_mm(2)], 'k-', 'LineWidth', 1.5);
t = [-3 3];
g7 = plot(-w_mm_n(2)*t, w_mm_n(1)*t, 'c-', 'LineWidth', 2);
for sgn = [-1 1]   % bandes de marge
    p0 = sgn * delta * w_mm_n;
    g8 = plot(p0(1) - w_mm_n(2)*t, p0(2) + w_mm_n(1)*t, 'c:', 'LineWidth', 1.2);
end
g9 = plot(-wW(2)*t, wW(1)*t, 'm--', 'LineWidth', 2);
xlim([-3 3]); ylim([-3 3]); xlabel('x(n)'); ylabel('x(n-1)');
title(sprintf('Enveloppe convexe de E+ et E-, \\delta = %.4f', delta));
legend([g1 g2 g3 g4 g5 g6 g7 g8 g9], ...
       {'conv(E+)', 'conv(E-)', 'Centres E+', 'Centres E-', 'Centres de support', ...
        'x* (point le plus proche de 0)', 'Hyperplan max-marge', 'Bandes de marge', ...
        'Hyperplan de Wiener (\sigma^2=0.2)'}, 'Location', 'eastoutside');
sauver('fig2_hyperplans_comparaison.png');

%% ---------------------------------------------------- 4. TEB en fonction du RSB
fprintf('\n===== 4. TEB = f(RSB) =====\n');
RSB_dB = 0:2:20;
N_teb  = 1e5;
nR = numel(RSB_dB);
T   = nan(nR, 4);     % simulation : Bayes, Wiener, MaxMarge, MinTEB
Tth = nan(nR, 3);     % theorie    : Wiener, MaxMarge, MinTEB
for i = 1:nR
    s2 = (c.' * c) / 10^(RSB_dB(i) / 10);       % RSB = 10 log10(c'Rd c / sigma2), Rd=I
    [X, d_ref] = generer_observations(c, P, N_teb, s2);
    w_w  = filtre_wiener(CC, s2);
    w_mt = hyperplan_min_teb(X_plus, s2, [w_w, w_mm_n]);
    d_B  = classifieur_bayes(X, X_plus, X_minus, s2);
    T(i, :)   = [mean(d_B ~= d_ref), teb_sim(w_w, X, d_ref), ...
                 teb_sim(w_mm_n, X, d_ref), teb_sim(w_mt, X, d_ref)];
    Tth(i, :) = [teb_lineaire_theorique(w_w, X_plus, s2), ...
                 teb_lineaire_theorique(w_mm_n, X_plus, s2), ...
                 teb_lineaire_theorique(w_mt, X_plus, s2)];
end
fprintf(' RSB(dB) |  Bayes    Wiener   MaxMarge  MinTEB   (simulation, N=%d)\n', N_teb);
for i = 1:nR
    fprintf(' %6.1f  | %8.5f %8.5f %8.5f %8.5f\n', RSB_dB(i), T(i, :));
end
T(T == 0) = NaN;   % pour l'echelle logarithmique

figure(3); clf;
semilogy(RSB_dB, T(:, 1), 'g-o', 'LineWidth', 2); hold on;
semilogy(RSB_dB, T(:, 2), 'm-s', 'LineWidth', 2);
semilogy(RSB_dB, T(:, 3), 'c-^', 'LineWidth', 2);
semilogy(RSB_dB, T(:, 4), 'k-d', 'LineWidth', 1.5);
semilogy(RSB_dB, Tth(:, 1), 'm:', 'LineWidth', 1.2);
semilogy(RSB_dB, Tth(:, 2), 'c:', 'LineWidth', 1.2);
grid on; xlabel('RSB (dB)'); ylabel('TEB');
title(sprintf('TEB en fonction du RSB, c=[%s], P=%d', num2str(c.', '%g '), P));
legend('Bayes (MAP)', 'Wiener', 'Hyperplan max-marge', 'Hyperplan min-TEB', ...
       'Wiener (theorique)', 'Max-marge (theorique)', 'Location', 'southwest');
sauver('fig3_teb_rsb.png');

%% ---------------------------------------------- 5. Influence du canal c
fprintf('\n===== 5. INFLUENCE DU CANAL (RSB = 10 dB, P = M) =====\n');
canaux = {[1; 0.5], [1; 0.9], [0.5; 1], [1; -0.5], [1; 0.5; 0.2], [0.4048; 0.8164; 0.4084]};
RSB_test = 10;  N_c = 5e4;
fprintf(' %-22s | sigma2 | delta   | separable | marge W | TEB Bayes | TEB Wiener | TEB MaxMarge\n', 'c');
for i = 1:numel(canaux)
    ci = canaux{i};  Pi = length(ci);
    [CCi, Xpi, Xmi] = centres_classes(ci, Pi);
    s2 = (ci.' * ci) / 10^(RSB_test / 10);
    [wmi, di] = hyperplan_max_marge(Xpi);
    [Xi, di_ref] = generer_observations(ci, Pi, N_c, s2);
    wWi = filtre_wiener(CCi, s2);
    teb_B = mean(classifieur_bayes(Xi, Xpi, Xmi, s2) ~= di_ref);
    teb_W = teb_sim(wWi, Xi, di_ref);
    if di > 1e-3, teb_M = teb_sim(wmi / di, Xi, di_ref); else, teb_M = NaN; end
    fprintf(' %-22s | %6.3f | %7.4f | %-9s | %7.4f | %9.5f | %10.5f | %10.5f\n', ...
        mat2str(ci.'), s2, di, mat2str(di > 1e-3), marge(wWi, Xpi), teb_B, teb_W, teb_M);
end

figure(4); clf; set(gcf, 'Position', [50 50 1100 480]);
subplot(1, 2, 1); tracer_classes_2D([1; 0.5], 0.1, 2000, 3, true);
subplot(1, 2, 2); tracer_classes_2D([0.5; 1], 0.1, 2000, 3, true);
sauver('fig4_separable_vs_non_separable.png');

%% ------------------------------------------------ 6. Montee en dimension M = 3
c3 = [1; 0.5; 0.2];
figure(5); clf;
tracer_classes_3D(c3, 0.05, 1500, 2.5);
sauver('fig5_3D_M3.png');

[CC3, Xp3, Xm3] = centres_classes(c3, 3);
[wm3, d3] = hyperplan_max_marge(Xp3);
fprintf('\nM=3, P=3 : delta = %.4f, marge Wiener(0.05) = %.4f\n', d3, ...
        marge(filtre_wiener(CC3, 0.05), Xp3));

%% ------------------------------------------ 7. M > 3 : reduction de dimension
c5 = [1; 0.5; 0.2; 0.1; 0.05];  M5 = length(c5);  P5 = M5;
s2 = (c5.' * c5) / 10^(10/10);
[CC5, Xp5, Xm5] = centres_classes(c5, P5);
[wm5, d5] = hyperplan_max_marge(Xp5);
[X5, d5_ref] = generer_observations(c5, P5, 20000, s2);
wW5 = filtre_wiener(CC5, s2);
fprintf('M=5, P=5 : delta = %.4f, separable = %s, marge Wiener = %.4f\n', ...
    d5, mat2str(d5 > 1e-3), marge(wW5, Xp5));
fprintf('TEB (RSB=10 dB) : Bayes = %.5f, Wiener = %.5f, MaxMarge = %.5f\n', ...
    mean(classifieur_bayes(X5, Xp5, Xm5, s2) ~= d5_ref), teb_sim(wW5, X5, d5_ref), ...
    teb_sim(wm5 / d5, X5, d5_ref));

ns = 1500;                                  % sous-echantillon pour t-SNE
Xs = X5(:, 1:ns);  ds = d5_ref(1:ns);
Xall = [Xs, Xp5, Xm5];                      % on projette aussi les centres
if exist('tsne', 'file')                    % Statistics and Machine Learning Toolbox
    Y = tsne(Xall.', 'NumDimensions', 2);
    nom_methode = 't-SNE';
else                                        % repli : ACP
    Xc = bsxfun(@minus, Xall, mean(Xall, 2));
    [U, ~, ~] = svd(Xc, 'econ');
    Y = (U(:, 1:2).' * Xc).';
    nom_methode = 'ACP (tsne indisponible)';
end
n1 = ns; n2 = ns + size(Xp5, 2);
figure(6); clf; hold on; grid on;
plot(Y(find(ds == 1),  1), Y(find(ds == 1),  2), 'b.', 'MarkerSize', 6);
plot(Y(find(ds == -1), 1), Y(find(ds == -1), 2), 'r.', 'MarkerSize', 6);
plot(Y(n1+1:n2, 1),   Y(n1+1:n2, 2),   'k+', 'MarkerSize', 8);
plot(Y(n2+1:end, 1),  Y(n2+1:end, 2),  'ko', 'MarkerSize', 6);
legend('E+ bruite', 'E- bruite', 'Centres E+', 'Centres E-', 'Location', 'best');
title(sprintf('%s, c=[%s], P=%d, RSB = 10 dB', nom_methode, num2str(c5.', '%g '), P5));
sauver('fig6_tsne_M5.png');

fprintf('\nTermine. Figures dans ./figures\n');
