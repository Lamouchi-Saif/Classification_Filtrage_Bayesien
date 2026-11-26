clear;
clc;
close all;

rng(42, "twister");

c = [1; 0.5];
M = length(c);
P = 2;
N = 100;

CC = zeros(P, P + M - 1);

for k = 1:P
    CC(k, k:k+M-1) = c';
end

fprintf("Matrice de convolution CC :\n");
disp(CC);

previous_symbols = [
     1   1  -1  -1
     1  -1   1  -1
];

D_plus = [ones(1, 4); previous_symbols];
D_minus = [-ones(1, 4); previous_symbols];

X_plus = CC * D_plus;
X_minus = CC * D_minus;

fprintf("Observations non bruitees de E+ :\n");
disp(X_plus);

fprintf("Observations non bruitees de E- :\n");
disp(X_minus);

if ~exist("../figures", "dir")
    mkdir("../figures");
end

% Figure 1 : motifs non bruites
figure(1);
plot(X_plus(1,:), X_plus(2,:), "bo", ...
     "MarkerSize", 9, "LineWidth", 2);
hold on;
plot(X_minus(1,:), X_minus(2,:), "r+", ...
     "MarkerSize", 10, "LineWidth", 2);
grid on;
axis equal;
xlabel("x(n)");
ylabel("x(n-1)");
title("Classes E+ et E- sans bruit");
legend("Classe E+", "Classe E-", "Location", "best");
print(gcf, "../figures/01_classes_sans_bruit.png", "-dpng", "-r180");

% Variances du bruit gaussien a tester
variances = [0.01, 0.2, 1.0];

% Figures 2, 3 et 4 : classes avec bruit
for q = 1:length(variances)
    sigma2 = variances(q);

    noisy_plus = [];
    noisy_minus = [];

    for k = 1:4
        bruit_plus = sqrt(sigma2) * randn(P, N);
        bruit_minus = sqrt(sigma2) * randn(P, N);

        noisy_plus = [noisy_plus, X_plus(:,k) + bruit_plus];
        noisy_minus = [noisy_minus, X_minus(:,k) + bruit_minus];
    end

    figure(q + 1);
    plot(noisy_plus(1,:), noisy_plus(2,:), "b.", "MarkerSize", 8);
    hold on;
    plot(noisy_minus(1,:), noisy_minus(2,:), "r.", "MarkerSize", 8);
    plot(X_plus(1,:), X_plus(2,:), "ko", ...
         "MarkerSize", 8, "LineWidth", 1.5);
    plot(X_minus(1,:), X_minus(2,:), "ks", ...
         "MarkerSize", 8, "LineWidth", 1.5);
    grid on;
    axis equal;
    xlabel("x(n)");
    ylabel("x(n-1)");
    title(sprintf("Classes avec bruit gaussien, variance = %.2f", sigma2));
    legend("E+ bruitee", "E- bruitee", ...
           "Centres E+", "Centres E-", "Location", "best");

    if q == 1
        filename = "../figures/02_classes_bruit_variance_0_01.png";
    elseif q == 2
        filename = "../figures/03_classes_bruit_variance_0_2.png";
    else
        filename = "../figures/04_classes_bruit_variance_1_0.png";
    end

    print(gcf, filename, "-dpng", "-r180");
end

% Figure 5 : comparaison des trois niveaux de bruit
figure(5);

for q = 1:length(variances)
    sigma2 = variances(q);
    noisy_plus = [];
    noisy_minus = [];

    for k = 1:4
        noisy_plus = [noisy_plus, X_plus(:,k) + sqrt(sigma2)*randn(P,N)];
        noisy_minus = [noisy_minus, X_minus(:,k) + sqrt(sigma2)*randn(P,N)];
    end

    subplot(1, 3, q);
    plot(noisy_plus(1,:), noisy_plus(2,:), "b.", "MarkerSize", 5);
    hold on;
    plot(noisy_minus(1,:), noisy_minus(2,:), "r.", "MarkerSize", 5);
    plot(X_plus(1,:), X_plus(2,:), "ko", "MarkerSize", 6);
    plot(X_minus(1,:), X_minus(2,:), "ks", "MarkerSize", 6);
    grid on;
    axis equal;
    xlabel("x(n)");
    ylabel("x(n-1)");
    title(sprintf("\\sigma^2 = %.2f", sigma2));
end

print(gcf, "../figures/05_comparaison_niveaux_bruit.png", "-dpng", "-r180");

fprintf("\nFigures enregistrees dans le dossier figures.\n");
fprintf("Simulation terminee.\n");
