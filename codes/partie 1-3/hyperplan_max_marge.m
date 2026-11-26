function [w, delta, lambda] = hyperplan_max_marge(X_plus, nb_iter)
% HYPERPLAN_MAX_MARGE  Hyperplan separateur a marge maximale (sans toolbox).
%
%   [w, delta, lambda] = hyperplan_max_marge(X_plus, nb_iter)
%
%   Idee geometrique : E- = -E+ (symetrie), donc l'hyperplan optimal passe par
%   l'origine. E+ et E- sont lineairement separables  <=>  0 n'appartient pas
%   a l'enveloppe convexe de E+. L'hyperplan de marge maximale est
%   perpendiculaire au point de l'enveloppe convexe le plus proche de 0 :
%
%        x* = argmin_{x in conv(X_plus)} ||x||   = X_plus * lambda*
%        w  = x*,   delta = ||x*||  (= marge geometrique, distance de 0 a conv)
%
%   Probleme resolu : min_lambda ||X_plus*lambda||^2  s.c. lambda>=0, sum=1
%   par gradient projete sur le simplexe (projection exacte, tri).
%
%   w      : normale (non normalisee, ||w|| = delta). Direction : w/norm(w).
%   delta  : degre de separabilite lineaire (delta ~ 0  =>  non separable).
%   lambda : poids des centres ; lambda>0 <=> "vecteurs de support".

    if nargin < 2, nb_iter = 20000; end

    n = size(X_plus, 2);
    G = X_plus.' * X_plus;
    L = 2 * max(eig(G));            % constante de Lipschitz du gradient
    lambda = ones(n, 1) / n;

    for it = 1:nb_iter
        lambda = proj_simplexe(lambda - (2*G*lambda) / L);
    end

    w = X_plus * lambda;
    delta = norm(w);
end

function x = proj_simplexe(v)
% Projection euclidienne de v (colonne) sur {x>=0, sum(x)=1}
    n = numel(v);
    u = sort(v, 'descend');
    css = cumsum(u) - 1;
    k = (1:n).';
    rho = find(u - css ./ k > 0, 1, 'last');
    theta = css(rho) / rho;
    x = max(v - theta, 0);
end
