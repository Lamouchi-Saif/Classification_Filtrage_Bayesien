function [w, teb] = hyperplan_min_teb(X_plus, sigma2, w0_liste)
% HYPERPLAN_MIN_TEB  Hyperplan lineaire (par l'origine) de TEB theorique minimal.
%
%   [w, teb] = hyperplan_min_teb(X_plus, sigma2, w0_liste)
%
%   Minimise J(w) = TEB_th(w) (voir teb_lineaire_theorique) par Nelder-Mead
%   (fminsearch, MATLAB de base). w0_liste : colonnes = points de depart
%   (ex. [w_wiener, w_max_marge]). On garde le meilleur. Resultat normalise.
%   Contrairement a Wiener (minimise l'EQM) ou au max-marge (pire cas),
%   c'est le meilleur classifieur lineaire au sens du TEB pour ce sigma2.

    opts = optimset('TolX', 1e-10, 'TolFun', 1e-14, ...
                    'MaxIter', 4000, 'MaxFunEvals', 8000, 'Display', 'off');
    f_best = Inf;
    w = w0_liste(:, 1) / norm(w0_liste(:, 1));
    for j = 1:size(w0_liste, 2)
        w0 = w0_liste(:, j) / norm(w0_liste(:, j));
        [wj, fj] = fminsearch(@(v) cout(v, X_plus, sigma2), w0, opts);
        if fj < f_best
            f_best = fj;
            w = wj / norm(wj);
        end
    end
    teb = teb_lineaire_theorique(w, X_plus, sigma2);
end

function f = cout(v, X_plus, sigma2)
    if norm(v) < 1e-12
        f = 1e10;
    else
        f = log(teb_lineaire_theorique(v, X_plus, sigma2) + 1e-300);
    end
end
