function [CC, X_plus, X_minus] = centres_classes(c, P)
% CENTRES_CLASSES  Matrice de convolution et motifs non bruites de E+ et E-.
%
%   [CC, X_plus, X_minus] = centres_classes(c, P)
%
%   c : coefficients du canal (M x 1), M = length(c) quelconque
%   P : taille du vecteur regresseur x(n) = [x(n); ...; x(n-P+1)]
%
%   CC      : matrice de convolution, P x (P+M-1)
%   X_plus  : valeurs non bruitees de x(n) quand d(n)=+1, P x 2^(P+M-2)
%   X_minus : idem quand d(n)=-1
%
%   Modele : x(n) = CC * D(n) + v(n),  D(n) = [d(n); d(n-1); ...; d(n-P-M+2)]

    c = c(:);
    M = length(c);
    L = P + M - 1;          % taille de D(n)
    K = L - 1;              % nombre de symboles "precedents" d(n-1)...
    nb_etats = 2^K;

    CC = zeros(P, L);
    for k = 1:P
        CC(k, k:k+M-1) = c.';
    end

    % Toutes les configurations (+-1) des K symboles precedents
    prev = zeros(K, nb_etats);
    for i = 0:nb_etats-1
        for j = 1:K
            prev(j, i+1) = 2*bitget(i, j) - 1;
        end
    end

    D_plus  = [ ones(1, nb_etats); prev];
    D_minus = [-ones(1, nb_etats); prev];

    X_plus  = CC * D_plus;
    X_minus = CC * D_minus;
end
