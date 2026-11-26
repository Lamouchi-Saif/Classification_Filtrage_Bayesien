function tracer_classes_2D(c, sigma2, N, lim, afficher_legende)
% TRACER_CLASSES_2D  Cas P=2 : motifs, observations bruitees, frontiere de
% Bayes (P+ = P-), hyperplan de Wiener et hyperplan a marge maximale.
%   Trace dans les axes courants.

    P = 2;
    [CC, Xp, Xm] = centres_classes(c, P);
    [X, d_ref] = generer_observations(c, P, N, sigma2);

    wW = filtre_wiener(CC, sigma2);  wW = wW / norm(wW);
    [wm, delta] = hyperplan_max_marge(Xp);

    hold on; grid on;
    ip = (d_ref ==  1);
    im = (d_ref == -1);
    h1 = plot(X(1, ip), X(2, ip), 'b.', 'MarkerSize', 6);
    h2 = plot(X(1, im), X(2, im), 'r.', 'MarkerSize', 6);
    h3 = plot(Xp(1, :), Xp(2, :), 'k+', 'MarkerSize', 12, 'LineWidth', 2);
    h4 = plot(Xm(1, :), Xm(2, :), 'ko', 'MarkerSize', 9,  'LineWidth', 2);

    % Frontiere de Bayes : ensemble { x | |P+(x) - P-(x)| < eps } -> contour a 0
    [X1, X2] = meshgrid(linspace(-lim, lim, 200));
    [~, Pg] = classifieur_bayes([X1(:).'; X2(:).'], Xp, Xm, sigma2);
    DIFF = reshape(Pg(1, :) - Pg(2, :), size(X1));
    [~, h5] = contour(X1, X2, DIFF, [0 0], 'g', 'LineWidth', 2);

    t = [-lim lim];
    h6 = plot(-wW(2)*t, wW(1)*t, 'm--', 'LineWidth', 2);
    handles = [h1 h2 h3 h4 h5 h6];
    noms = {'E+ bruite', 'E- bruite', 'Centres E+', 'Centres E-', ...
            'Frontiere de Bayes', 'Hyperplan de Wiener'};

    if delta > 1e-3
        wn = wm / delta;
        h7 = plot(-wn(2)*t, wn(1)*t, 'c-.', 'LineWidth', 2);
        handles(end+1) = h7;
        noms{end+1} = 'Hyperplan max-marge';
    end

    axis equal; xlim([-lim lim]); ylim([-lim lim]);
    xlabel('x(n)'); ylabel('x(n-1)');
    title(sprintf('c=[%s], \\sigma_v^2=%.2f, \\delta=%.3f', ...
          num2str(c(:).', '%g '), sigma2, delta));
    if afficher_legende
        legend(handles, noms, 'Location', 'best');
    end
end
