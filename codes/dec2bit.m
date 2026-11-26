function[u]=dec2bit(nombre,p);
%p=fix(log(nombre)/log(2));
u=zeros(p+1,1);
if nombre>0,
for j=1:p+1,
    u(j)=fix(nombre/(2^(p+1-j)));
    nombre=nombre-u(j)*(2^(p+1-j));

end
end
