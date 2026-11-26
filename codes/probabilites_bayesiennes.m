
function P = probabilites_bayesiennes(X, X_plus, X_minus, sigma2)
% PROBABILITES_BAYESIENNES
% Calcule les probabilites a posteriori des classes E+ et E-.
%
% Entrees :
%   X       : observations, dimensions 2 x K
%   X_plus  : centres non bruites de E+, dimensions 2 x 4
%   X_minus : centres non bruites de E-, dimensions 2 x 4
%   sigma2  : variance du bruit gaussien
%
% Sortie :
%   P       : matrice 2 x K
%             premiere ligne : P+
%             deuxieme ligne : P-

    if sigma2 <= 0
        error('La variance sigma2 doit etre strictement positive.');
    end

    K = size(X, 2);

    P = zeros(2, K);

    for n = 1:K

        % Distances au carre entre l'observation et les centres
        dist_plus = sum((X_plus - X(:,n)).^2, 1);
        dist_minus = sum((X_minus - X(:,n)).^2, 1);

        % Logarithmes des poids gaussiens
        logw_plus = -dist_plus / (2 * sigma2);
        logw_minus = -dist_minus / (2 * sigma2);

        % Log-sum-exp pour eviter les sous-debordements numeriques
        max_plus = max(logw_plus);
        max_minus = max(logw_minus);

        logp_plus = max_plus + log(sum(exp(logw_plus - max_plus)));
        logp_minus = max_minus + log(sum(exp(logw_minus - max_minus)));

        % Normalisation : les constantes communes s'annulent
        max_global = max(logp_plus, logp_minus);

        a = exp(logp_plus - max_global);
        b = exp(logp_minus - max_global);

        P(1,n) = a / (a + b);
        P(2,n) = b / (a + b);

    end

end
