function [d_hat, Pb] = classifieur_bayes(X, X_plus, X_minus, sigma2)
% CLASSIFIEUR_BAYES  Probabilites a posteriori et decision MAP (vectorise).
%
%   [d_hat, Pb] = classifieur_bayes(X, X_plus, X_minus, sigma2)
%
%   X : P x N observations.  Pb(1,:) = P+(n),  Pb(2,:) = P-(n),  d_hat dans {+1,-1}.
%
%   P+(n) = sum_i exp(-||x-s_i^+||^2/(2 sigma2)) / [ sum_i (...)^+ + sum_i (...)^- ]
%   (priors egaux, constantes gaussiennes communes simplifiees ; log-sum-exp
%   pour la stabilite numerique). Equivalent vectorise de probabilites_bayesiennes.m

    nX = sum(X.^2, 1).';                                   % N x 1
    D_plus  = bsxfun(@plus, nX, sum(X_plus.^2, 1))  - 2 * (X.' * X_plus);
    D_minus = bsxfun(@plus, nX, sum(X_minus.^2, 1)) - 2 * (X.' * X_minus);

    lp = lse(-D_plus  / (2*sigma2));                       % N x 1
    lm = lse(-D_minus / (2*sigma2));

    Pplus = 1 ./ (1 + exp(lm - lp));
    Pb = [Pplus.'; 1 - Pplus.'];
    d_hat = 2 * (Pb(1, :) > Pb(2, :)) - 1;
end

function y = lse(A)
% log(sum(exp(A),2)) stable
    m = max(A, [], 2);
    y = m + log(sum(exp(bsxfun(@minus, A, m)), 2));
end
