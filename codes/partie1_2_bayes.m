
function partie1_2_bayes()
% ============================================================
% PARTIE 1.2 - CLASSIFICATION BAYESIENNE
% Calcul des probabilites, classification MAP, TEB et
% affichage de la frontiere de decision bayesienne.
%
% Modele :
% x(n) = somme(c(i)*d(n-i)) + v(n)
% v(n) ~ N(0, sigma2)
%
% Classes :
% E+ : d(n) = +1
% E- : d(n) = -1
%
% ============================================================

clc;
close all;

%% 1. PARAMETRES

c = [1; 0.5];       % Coefficients du canal
M = length(c);      % Longueur de la reponse impulsionnelle
P = 2;              % Dimension du vecteur d'observation
N = 10000;          % Nombre d'observations
sigma2 = 0.2;       % Variance du bruit gaussien

rng(42);            % Pour rendre les simulations reproductibles

% Nombre de symboles precedents necessaires pour construire
% le vecteur d'observation
K = P + M - 2;

% Nombre de configurations possibles des symboles precedents
nb_etats = 2^K;

if P ~= 2
    error('Ce script affiche la frontiere en 2D : choisir P = 2.');
end

if sigma2 <= 0
    error('La variance sigma2 doit etre strictement positive.');
end

if K > 16
    error('Trop de configurations. Reduire P ou M pour cette version.');
end

fprintf('============================================\n');
fprintf(' CLASSIFICATION BAYESIENNE - PARTIE 1.2\n');
fprintf('============================================\n');
fprintf('M = %d, P = %d, N = %d\n', M, P, N);
fprintf('Variance du bruit sigma2 = %.4f\n', sigma2);

%% 2. CONSTRUCTION DE LA MATRICE DE CONVOLUTION CC

% CC est de dimension P x (P+M-1)
CC = zeros(P, P + M - 1);

for k = 1:P
    CC(k, k:k+M-1) = c';
end

fprintf('\nMatrice de convolution CC :\n');
disp(CC);

%% 3. GENERATION DES ETATS NON BRUITES DES DEUX CLASSES

% Chaque colonne represente une configuration possible
% des K = P+M-2 symboles precedents.
previous_symbols = zeros(K, nb_etats);

for i = 0:nb_etats-1
    for j = 1:K
        previous_symbols(j, i+1) = 2*bitget(i, j)-1;
    end
end

% Classe E+ : symbole courant d(n) = +1
D_plus = [ones(1, nb_etats); previous_symbols];

% Classe E- : symbole courant d(n) = -1
D_minus = [-ones(1, nb_etats); previous_symbols];

% Observations non bruitees des deux classes
X_plus = CC * D_plus;
X_minus = CC * D_minus;

% Regroupement des centres des deux classes
tous_centres = [X_plus, X_minus];
fprintf('\nCentres non bruites de la classe E+ :\n');
disp(X_plus);

fprintf('Centres non bruites de la classe E- :\n');
disp(X_minus);

%% 4. GENERATION DES OBSERVATIONS BRUITEES

% Symboles binaires aleatoires : -1 ou +1
% On genere suffisamment de symboles pour eviter
% les problemes d'initialisation du filtre.
d = 2*randi([0 1], 1, N + K) - 1;

% Signal recu : convolution du canal et du signal binaire
% puis ajout du bruit blanc gaussien
bruit = sqrt(sigma2) * randn(1, N + K);
x = filter(c, 1, d) + bruit;

% Construction du vecteur d'observation :
% X(:,n) = [x(t); x(t-1); ...; x(t-P+1)]
X = zeros(P, N);
d_reference = zeros(1, N);

for n = 1:N

    t = K + n;

    X(:,n) = x(t:-1:t-P+1);

    % Symbole courant associe a cette observation
    d_reference(n) = d(t);

end
% Indices des observations appartenant a chaque classe
indices_plus = find(d_reference == 1);
indices_minus = find(d_reference == -1);


%% 5. AFFICHAGE DES CLASSES ET OBSERVATIONS BRUITEES

figure(1);
clf;
hold on;

% Afficher seulement une observation sur 10 pour eviter
% de masquer les centres des classes
pas_affichage = 10;

indices_affiches_plus = indices_plus(1:pas_affichage:end);
indices_affiches_minus = indices_minus(1:pas_affichage:end);

% 1. Observations bruitees en premier (arriere-plan)
plot(X(1,indices_affiches_plus), ...
     X(2,indices_affiches_plus), ...
     'b*', 'MarkerSize', 4);

plot(X(1,indices_affiches_minus), ...
     X(2,indices_affiches_minus), ...
     'r*', 'MarkerSize', 4);

% 2. Centres non bruites en dernier (premier plan)
plot(X_plus(1,:), X_plus(2,:), ...
     'c+', 'MarkerSize', 16, 'LineWidth', 3);

plot(X_minus(1,:), X_minus(2,:), ...
     'mo', 'MarkerSize', 12, 'LineWidth', 3);

grid on;
axis equal;

xlabel('x(n)');
ylabel('x(n-1)');

title('Classes et observations en presence de bruit');

legend('Observations E+ bruitees', ...
       'Observations E- bruitees', ...
       'Centres E+', ...
       'Centres E-', ...
       'Location', 'best');

% Zoom automatique avec une marge autour des centres
xlim([min(tous_centres(1,:))-1, max(tous_centres(1,:))+1]);
ylim([min(tous_centres(2,:))-1, max(tous_centres(2,:))+1]);

hold off;


%% 6. CALCUL DES PROBABILITES BAYESIENNES

% P_bayes(1,n) = P+(n)
% P_bayes(2,n) = P-(n)

P_bayes = zeros(2, N);

for n = 1:N

    % Distances au carre entre l'observation et les centres E+
    dist_plus = sum(bsxfun(@minus, X_plus, X(:,n)).^2, 1);


    % Distances au carre entre l'observation et les centres E-
    dist_minus = sum(bsxfun(@minus, X_minus, X(:,n)).^2, 1);

    % Logarithmes des poids gaussiens
    logw_plus = -dist_plus / (2*sigma2);
    logw_minus = -dist_minus / (2*sigma2);

    % Calcul stable de log(sum(exp(...)))
    max_plus = max(logw_plus);
    max_minus = max(logw_minus);

    logp_plus = max_plus + ...
        log(sum(exp(logw_plus - max_plus)));

    logp_minus = max_minus + ...
        log(sum(exp(logw_minus - max_minus)));

    % Normalisation numerique
    max_global = max(logp_plus, logp_minus);

    a = exp(logp_plus - max_global);
    b = exp(logp_minus - max_global);

    P_bayes(1,n) = a / (a + b);
    P_bayes(2,n) = b / (a + b);

end

fprintf('\nExemple de probabilites pour les 5 premieres observations :\n');
disp(P_bayes(:,1:5));

% Verification de la somme des probabilites
erreur_normalisation = max(abs(sum(P_bayes,1) - 1));

fprintf('Erreur maximale de normalisation : %.12f\n', ...
    erreur_normalisation);

%% 7. CLASSIFICATION MAP

% Estimation du symbole d(n) par la methode MAP
d_MAP = -ones(1, N);

for n = 1:N

    if P_bayes(1,n) > P_bayes(2,n)
        d_MAP(n) = +1;
    else
        d_MAP(n) = -1;
    end

end

%% 8. CALCUL DU TAUX D'ERREUR BINAIRE (TEB)

% Une erreur est commise lorsque d_MAP est different
% du symbole reel d_reference.

nombre_erreurs = sum(d_MAP ~= d_reference);

TEB = nombre_erreurs / N;

fprintf('\n============================================\n');
fprintf(' RESULTATS DE LA CLASSIFICATION MAP\n');
fprintf('============================================\n');

fprintf('Nombre total d observations : %d\n', N);
fprintf('Nombre de classifications incorrectes : %d\n', ...
    nombre_erreurs);
fprintf('TEB = %.6f\n', TEB);
fprintf('TEB en pourcentage = %.4f %%\n', 100*TEB);

%% 9. DISCRETISATION DE L'ESPACE D'OBSERVATION

% Determination des limites de la grille a partir
% des centres non bruites des deux classes.

tous_centres = [X_plus, X_minus];

xmin = min(tous_centres(1,:));
xmax = max(tous_centres(1,:));

ymin = min(tous_centres(2,:));
ymax = max(tous_centres(2,:));

% Marge autour des classes
marge_x = max(0.5, 0.25*(xmax-xmin));
marge_y = max(0.5, 0.25*(ymax-ymin));

xmin = xmin - marge_x;
xmax = xmax + marge_x;

ymin = ymin - marge_y;
ymax = ymax + marge_y;

% Grille de discretisation
nb_points = 250;

x1 = linspace(xmin, xmax, nb_points);
x2 = linspace(ymin, ymax, nb_points);

[X1, X2] = meshgrid(x1, x2);

% Chaque colonne correspond a un point de la grille
X_grid = [X1(:)'; X2(:)'];

%% 10. CALCUL DES PROBABILITES SUR LA GRILLE

nb_grille = size(X_grid, 2);

P_grid = zeros(2, nb_grille);

for n = 1:nb_grille

    dist_plus = sum(bsxfun(@minus, X_plus, X_grid(:,n)).^2, 1);
    dist_minus = sum(bsxfun(@minus, X_minus, X_grid(:,n)).^2, 1);

    logw_plus = -dist_plus / (2*sigma2);
    logw_minus = -dist_minus / (2*sigma2);

    max_plus = max(logw_plus);
    max_minus = max(logw_minus);

    logp_plus = max_plus + ...
        log(sum(exp(logw_plus - max_plus)));

    logp_minus = max_minus + ...
        log(sum(exp(logw_minus - max_minus)));

    max_global = max(logp_plus, logp_minus);

    a = exp(logp_plus - max_global);
    b = exp(logp_minus - max_global);

    P_grid(1,n) = a / (a + b);
    P_grid(2,n) = b / (a + b);

end

% Difference entre les probabilites des deux classes
difference_P = P_grid(1,:) - P_grid(2,:);

% Remise sous forme de matrices 2D
DIFF = reshape(difference_P, size(X1));


%% 11. AFFICHAGE DE LA FRONTIERE BAYESIENNE

epsilon = 0.001;

masque_frontiere = abs(DIFF) < epsilon;

figure(2);
clf;
hold on;

% 1. Affichage de la frontiere theorique
contour(X1, X2, DIFF, [0 0], ...
        'k', 'LineWidth', 2);

% 2. Affichage des points proches de P+ = P-
plot(X1(masque_frontiere), ...
     X2(masque_frontiere), ...
     'g.', 'MarkerSize', 12);

% 3. Affichage d'un sous-ensemble des observations bruitees
pas_affichage = 10;

indices_affiches_plus = indices_plus(1:pas_affichage:end);
indices_affiches_minus = indices_minus(1:pas_affichage:end);

plot(X(1,indices_affiches_plus), ...
     X(2,indices_affiches_plus), ...
     'b*', 'MarkerSize', 4);

plot(X(1,indices_affiches_minus), ...
     X(2,indices_affiches_minus), ...
     'r*', 'MarkerSize', 4);

% 4. Centres non bruites au premier plan
plot(X_plus(1,:), X_plus(2,:), ...
     'c+', 'MarkerSize', 16, 'LineWidth', 3);

plot(X_minus(1,:), X_minus(2,:), ...
     'mo', 'MarkerSize', 12, 'LineWidth', 3);

grid on;
axis equal;

xlabel('x(n)');
ylabel('x(n-1)');

title(sprintf('Frontiere bayesienne : sigma2 = %.2f', sigma2));

legend('Frontiere P+ = P-', ...
       'Points |P+ - P-| < epsilon', ...
       'Observations E+ bruitees', ...
       'Observations E- bruitees', ...
       'Centres E+', ...
       'Centres E-', ...
       'Location', 'best');

% Zoom autour des classes
xlim([xmin xmax]);
ylim([ymin ymax]);

hold off;

fprintf('\nNombre de points proches de la frontiere : %d\n', ...
        sum(masque_frontiere(:)));
