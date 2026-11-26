function [X, d_ref, x] = generer_observations(c, P, N, sigma2)
% GENERER_OBSERVATIONS  Simule N vecteurs regresseurs bruites.
%
%   [X, d_ref] = generer_observations(c, P, N, sigma2)
%
%   X     : P x N, colonne n = [x(t); x(t-1); ...; x(t-P+1)]
%   d_ref : 1 x N, symbole d(t) associe a chaque colonne
%   x     : signal recu complet (utile pour debogage)
%
%   x(n) = sum_i c(i) d(n-i) + v(n),  v ~ N(0, sigma2),  d(n) iid dans {+1,-1}

    c = c(:);
    M = length(c);
    K = P + M - 2;

    d = 2*randi([0 1], 1, N + K) - 1;
    x = filter(c, 1, d) + sqrt(sigma2) * randn(1, N + K);

    X = zeros(P, N);
    d_ref = zeros(1, N);
    for n = 1:N
        t = K + n;
        X(:, n) = x(t:-1:t-P+1).';
        d_ref(n) = d(t);
    end
end
