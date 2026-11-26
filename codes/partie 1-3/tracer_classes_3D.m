function tracer_classes_3D(c, sigma2, N, lim)
% TRACER_CLASSES_3D  Cas M=3, P=3 : nuages de points, centres, plans de
% Wiener et de marge maximale. Les plans sont parametres par une base
% orthonormee du plan (null(w')), ce qui evite de diviser par w(3).

    P = 3;
    [CC, Xp, Xm] = centres_classes(c, P);
    [X, d_ref] = generer_observations(c, P, N, sigma2);

    wW = filtre_wiener(CC, sigma2);  wW = wW / norm(wW);
    [wm, delta] = hyperplan_max_marge(Xp);

    hold on; grid on;
    ip = (d_ref ==  1);
    im = (d_ref == -1);
    h1 = plot3(X(1, ip), X(2, ip), X(3, ip), 'b.', 'MarkerSize', 6);
    h2 = plot3(X(1, im), X(2, im), X(3, im), 'r.', 'MarkerSize', 6);
    h3 = plot3(Xp(1, :), Xp(2, :), Xp(3, :), 'k+', 'MarkerSize', 12, 'LineWidth', 2);
    h4 = plot3(Xm(1, :), Xm(2, :), Xm(3, :), 'ko', 'MarkerSize', 9,  'LineWidth', 2);

    [A, B] = meshgrid(linspace(-lim, lim, 12));
    U = null(wW.');
    hW = surf(U(1,1)*A + U(1,2)*B, U(2,1)*A + U(2,2)*B, U(3,1)*A + U(3,2)*B, ...
              'FaceColor', 'm', 'FaceAlpha', 0.25, 'EdgeColor', 'none');
    handles = [h1 h2 h3 h4 hW];
    noms = {'E+ bruite', 'E- bruite', 'Centres E+', 'Centres E-', 'Plan de Wiener'};

    if delta > 1e-3
        wn = wm / delta;
        U2 = null(wn.');
        hM = surf(U2(1,1)*A + U2(1,2)*B, U2(2,1)*A + U2(2,2)*B, U2(3,1)*A + U2(3,2)*B, ...
                  'FaceColor', 'c', 'FaceAlpha', 0.25, 'EdgeColor', 'none');
        handles(end+1) = hM;
        noms{end+1} = 'Plan max-marge';
    end

    axis equal; xlim([-lim lim]); ylim([-lim lim]); zlim([-lim lim]);
    xlabel('x(n)'); ylabel('x(n-1)'); zlabel('x(n-2)');
    title(sprintf('c=[%s], \\sigma_v^2=%.2f, \\delta=%.3f', ...
          num2str(c(:).', '%g '), sigma2, delta));
    legend(handles, noms, 'Location', 'best');
    view(3);
end
