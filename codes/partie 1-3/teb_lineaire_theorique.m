function teb = teb_lineaire_theorique(w, X_plus, sigma2)
% TEB_LINEAIRE_THEORIQUE  TEB exact d'un classifieur lineaire d_hat = sign(w'x).
%
%   Pour un centre s de E+, w'x ~ N(w's, sigma2*||w||^2) : l'erreur vaut
%   Q( w's / (sigma*||w||) ). Par symetrie (E- = -E+) et equiprobabilite :
%       TEB = moyenne sur les centres s de E+ de Q( w's / (sigma*||w||) )
%   avec Q(x) = 0.5*erfc(x/sqrt(2)).

    Q = @(x) 0.5 * erfc(x / sqrt(2));
    w = w(:);
    teb = mean(Q((w.' * X_plus) / (norm(w) * sqrt(sigma2))));
end
