m=fscanfMat("Kangourous.txt");
size(m);
m=gsort(m,"lr","i");
plot(m(:,1)',m(:,2)');

//a = conv(x,y)/Variance(X) => corr(x,y,1) / variance(x)
//b = moyenne y - moyenne x*a 

[a,b]=reglin(x,y);

clf() //effacer notre fenetre
plot2d(x,y,-2);
plot2d(x,a*x+b);

coXY = corr(x,y,1);
Vx   = variance(x);

Aa = coXY/Vx;
Bb = mean(y)- Aa*mean(x);


A = [x',ones(x')];
Z = A\y'

//la difference entre le points corrélation et le point réel pour la somme permet de savoir si il y a une forte corrélation entre la longeur et la largeur d'un nez du kangourou
