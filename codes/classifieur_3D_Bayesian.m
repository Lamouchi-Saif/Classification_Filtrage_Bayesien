% ============ code classififcation Bayésienne (R. Amara) ============
% ============ cas de la déconvolution ===============================

close all
clear all

%===========================================
c=[1;0.5;0.2]; % coefficients de la RI du canal à tester
%c=[0.4048;0.8164;0.4084];
M=length(c);
sigv=0.01;   % variance bruit additif   pour .001  1e-3    pour 0.01   1e-30
N=10^4;   % nombre de données
P=3;


cas=1; %  1 pour wiener, 2 for Bayesian classifier

%===========================================
%======== génération de d et x ===================================
d=sign(randn(1,N));  % d : séquence binaire générée de longueur N
d_buffer=zeros(M,1); % Buffer
for n=1:N,
    d_buffer=[d(n);d_buffer(1:M-1)];
    x(n)=c'*d_buffer+sqrt(sigv)*randn; % entrée récepteur ( convolution avec la RI du canal)
end


% ============ filtre de wiener pour la déconvolution =====
CC=[c' 0 0;0 c' 0;0 0 c'];
%CC=[c' 0 0 0;0 c' 0 0;0 0 c' 0;0 0 0 c'];
c_inter=[c(1);0;0];
%c_inter=[c(1);0;0;0];
wopt=inv(CC*CC'+sigv*eye(P))*c_inter;

%===== filtrage de Wiener pour la déconvolution ================ 
% ==== et visualisation de vecteur observation régresseur x(n)==

x_buffer=zeros(P,1);
x_buffer_mat1=[];
x_buffer_mat0=[];
for n=1:N,
   x_buffer=[x(n);x_buffer(1:P-1)];
   x_buffer_mat(:,n)=x_buffer;
   y(n)=wopt'*x_buffer;
%=========================================================
if mod(n,10)==0,    
if d(n)==1,
    plot3(x_buffer(1),x_buffer(2),x_buffer(3),'ok');hold on;pause(0.0001)
x_buffer_mat1=[x_buffer_mat1 x_buffer];
else,
    plot3(x_buffer(1),x_buffer(2),x_buffer(3),'*r');hold on;pause(0.0001)
    x_buffer_mat0=[x_buffer_mat0 x_buffer];
    
end
%==========================================================
end    
end
%plot(x_buffer_mat0(1,:),x_buffer_mat0(2,:),'*r')
grid on
disp('ok pour affichage des patterns pour les deux classes')

if cas==1; % surface de décision obtenue pour le filtre de wiener
xx=-2:0.1:2;
for i=1:length(xx); 
   for j=1:length(xx);
        for l=1:length(xx);

y=wopt'*[xx(i);xx(j);xx(l)];
if abs(y)<2e-2,
plot3(xx(i),xx(j),xx(l),'*g');hold on;pause(0.000001)            
end
        end
    end
end
end

if cas==2; % égalisation du canal comme par classif. Bayésienne

% noise free channel states generation
M=3;P=3;NN=2^(M+P-2);
for i=0:NN-1,
d_L(:,i+1)=2*dec2bit(i,M+P-3)-1;
end

d_L_classe0=[-ones(1,NN);d_L]; 
d_L_classe1=[ones(1,NN);d_L];

x_nf_classe0=CC*d_L_classe0;
x_nf_classe1=CC*d_L_classe1;

for i=1:N,
    X_n=kron(x_buffer_mat(:,i),ones(1,NN));
    diff_classe0=X_n-x_nf_classe0;
    diff_classe1=X_n-x_nf_classe1;
    weights_classe0=exp(-1/(2*sigv)*sum(diff_classe0.^2));
    weights_classe1=exp(-1/(2*sigv)*sum(diff_classe1.^2));
    sumweights=sum(weights_classe0)+sum(weights_classe1);
    norm_weights_classe0=weights_classe0/sumweights;
    norm_weights_classe1=weights_classe1/sumweights;

    dhat(i)=-1;
    P0=sum(norm_weights_classe0);
    P1=sum(norm_weights_classe1);
   if P1>P0, 
       dhat(i)=+1;
   end 
   disp('[P1 P0]=') 
   [P1 P0]
   dhat(i)
   d(i)
   pause
    %if abs(P1-P0)<0.1
    %plot3(x_buffer_mat(:,1),x_buffer_mat(:,2),x_buffer_mat(:,3),'*g');hold on;pause(0.0001)    
    %end
    
    
    
end
end
