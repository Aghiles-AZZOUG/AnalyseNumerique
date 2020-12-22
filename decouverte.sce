//fonction racine
X=[0:1:10];
racine=sqrt(X);
plot(X,racine(X));

/*
clf: clear or reset the current graphic figure (window) to default values
scf: set the current graphic figure (window), ouvrir nvlle fenetre pr ploter des figures
*/
//clf)) : effacer le graphique
//scf(2): set current figure

//fonction carré
carre=X^2;

// même graphe
plot(X,racine(X),X,carre(X));
//2 graphes sur le même plan avec subplot
subplot(1,2,1);
plot(X,racine(X),'r');
subplot(1,2,2);
plot(X,carre(X),'b');


// export file
xs2png(0, 'file_name');

//la fonction qui somme deux éléments
function [x]=somme(a,b)
    x=a+b;
endfunction

// matrice du TD1
A=[1 -1 2;4 -6 12;1 -5 12];

//Multiplication matricielle A*A
B=(A+A)*A;

//Multiplication terme à terme A.*A
B=A.*A

det(B);//determinant
C=inv(B); //inverse

// 1ere ligne de B
l=B(1:1,1:3);

//somme d'une ligne (utiliser sum)
function [x]=somme_vec(M)
    maxl=size(M,2);
    somme=0;
    for i= 1:maxl
        somme=somme+M(i);
    end
    x=somme;
endfunction

somme_vec(l);


//cosinus
Z=cos(l);

// Exercice 2
// ouvrir le fichier, affecter les valeurs à my_mat et fermer le fichier
Fichier=mopen('C:\Users\azzou\Documents\scilab\kangourou.txt','rt');
my_mat=mfscanf(-1,Fichier,'%f \t %f');
mclose(Fichier);

// method 2
my_mat=csvRead("Kang.txt", "\t");

//longueur=f(largeur)
longueur=my_mat(1:size(my_mat,1),1:1);
largeur=my_mat(1:size(my_mat,1),2:2);

//tri 
[H]=gsort(my_mat,'lr' );
largeur=H(1:size(my_mat,1),2:2);
longueur=H(1:size(my_mat,1),1:1);
 plot(longueur, largeur);
 
 //regression linéaire
 [a, b] = reglin(H(:,1)', H(:,2)');
 X=[0:900];
 plot(X,a*X+b);
 
 //donnees brutes
Fichier=mopen('C:\Users\khali\Documents\kangourou.txt','rt');
my_matt=mfscanf(-1,Fichier,'%f \t %f');
mclose(Fichier);

longueur=my_matt(1:size(my_matt,1),1:1);
largeur=my_matt(1:size(my_matt,1),2:2);

plot(longueur, largeur,"r");
[a, b] = reglin(my_matt(:,1)', my_matt(:,2)');
X=[0:900];
plot(X,a*X+b,"cyan");

 
 //titre
 xtitle("Titre du graphique","Longueur","Largeur");
 
 //5
 a=(corr(H(:,1)',H(:,2)))/variance(H(:,1)'); 
 
 // var-cov
 a=cov(my_mat(:,1),my_mat(:,2)) / variance(H(:,1));
 b= mean(H(:,2)) - a * mean (H(:,1));
 
 
//Question 5
[in,My] = variance(H(:,2));
[v,Mx] = variance(H(:,1));
c = cov(H(:,1),H(:,2),1);
a = c/v;
b = My - a*Mx;
subplot(2,1,1);
plot(H(:,1)', a*H(:,1)'+b, "r");
subplot(2,1,2);
[a, b] = reglin(H(:,1)', H(:,2)');
plot(H(:,1),a*H(:,1)+b,"b");

somme_vec(l);




//TP2
//1

// fonction auxiliaire
function [M]=matrice_ck(A,C,R,n)
    M(1,1)=C;
    M(2,1)=R;
    for i=2:n
        M(1,i)=A(1,1) * M(1,i-1) + A(1,2) * M(2,i-1);
        M(2,i)=A(2,1) * M(1,i-1) + A(2,2) * M(2,i-1);
    end
endfunction
//
A=[ 0.95 0.2 ; 0.05 -0.01];
B=matrice_ck(A,475,25,20);
T=[1:20];
plot(T,B);



// version ro
function [r] = generateData(n)
    rats = 0:1:n
    chouettes = 0:1:n
    rats(1) = 10
    chouettes(1) = 13
    for i=2:1:(n+1)
        chouettes(i) = chouettes(i-1) * 0.5 + rats(i-1) * 0.4
        rats(i) = chouettes(i-1) * (-0.104) + rats(i-1) * 1.1
    end
    r = [rats ; chouettes]
endfunction


function [] = display(data, n)
    abscisse = 0:1:n
    ratio = 0:1:n
    for i=1:1:(n+1)
        ratio(i) = data(1,i)/data(2,i)
    end
    scf(0);
    clf(); // efface la fenêtre graphique
    plot2d(abscisse,data(1,:), style=20);
    plot2d(abscisse,data(2,:), style=13);
    legend("rats", "chouettes")
   // f = scf(1);
    scf(1);
    clf()
    plot2d(abscisse,ratio, style=10);
    legend("ratio(rats/chouettes)")
    scf(2);
    clf()
    [a,b] = reglin(data(2,15:20),data(1,15:20))
    line = 7:0.5:13
    abline = 7:0.5:13
    line = line+b
    line = line.*a
    plot2d(data(2,:), data(1,:));
    plot2d(abline,line,13)
endfunction


function [] = gdisplay(n)
    data = generateData(n)
    display(data, n)
endfunction

//2
X=B(1,10:20);
[a, b] = reglin(B(1,10:20), B(2,10:20));
plot(B(1,:),B(2,:));
plot(X,a*X+b,"red");

//3
//la pente est (1+t), démo en cours

Rka=B(2,2:20);
Rkb=B(2,1:19);
subplot(2,2,1);
plot(Rkb,Rka,"blue");


Rkar=B(2,2:20);
Rkbr=B(2,1:19);
subplot(2,2,2);
plot(Rkbr,Rkar,"red");

[a, b] = reglin(Rkb(1,10:19), Rka(1,10:19));
subplot(2,2,3);
plot(X,a*X+b,"green");

[c,d]=reglin(Rkbr(1,10:19), Rkar(1,10:19));
subplot(2,2,4);
plot(X,c*X+d,"cyan");



Rka=B(2,2:20);
Rkb=B(2,1:19);
subplot(2,2,1);
plot(Rkb,Rka,"blue");


Rkar=B(2,2:20);
Rkbr=B(2,1:19);
subplot(2,2,2);
plot(Rkbr,Rkar,"red");

[a, b] = reglin(Rkb(1,10:19), Rka(1,10:19));
subplot(2,2,3);
plot(X,a*X+b,"green");

[c,d]=reglin(Rkbr(1,10:19), Rkar(1,10:19));
subplot(2,2,4);
plot(X,c*X+d,"cyan");
//4
// Rk=(1+t/100)^k Rk-1
//ln Rk=k ln(1+ t/100) +ln R0

plot(X,log(1+ X/100)*X+log(10),"blue");


//suite
//5
L=matrice_ck(A,10,17,20);
M=matrice_ck(A,5,10,20);
N=matrice_ck(A,20,9,20);
O=matrice_ck(A,14,4,20);


//reprise de la question 2
subplot(2,2,1);
plot(T,L);
subplot(2,2,2);
plot(T,M);
subplot(2,2,3);
plot(T,N);
subplot(2,2,4);
plot(T,O);

subplot(2,2,1);
X=L(1,10:20);
[a, b] = reglin(L(1,10:20), L(2,10:20));
plot(L(1,:),L(2,:),"cyan");
plot(X,a*X+b,"red");

subplot(2,2,2);
X=M(1,10:20);
[a, b] = reglin(M(1,10:20), M(2,10:20));
plot(M(1,:),M(2,:),"cyan");
plot(X,a*X+b,"red");

subplot(2,2,3);
X=N(1,10:20);
[a, b] = reglin(N(1,10:20), N(2,10:20));
plot(N(1,:),N(2,:),"cyan");
plot(X,a*X+b,"red");

subplot(2,2,4);
X=O(1,10:20);
[a, b] = reglin(O(1,10:20), O(2,10:20));
plot(O(1,:),O(2,:),"cyan");
plot(X,a*X+b,"red");


////////////////////
A = [0.5, 0.4;
    -0.104, 1.1;]

V = [13;10]

//Q1
n = 100
for i = 1:n
    resultat(1:2,i) = A*V
    V = resultat(1:2, i)
end

FL = resultat(1,:)
plot(FL)

SL = resultat(2,:)
plot(SL)

//Q2
[a,b] = reglin(SL,FL)
x = 1:100
plot(x, a*x+b)

//Q3
x = resultat(2, 1:99)
y = resultat(2, 2:100)
plot(x,y)
[c,d] = reglin(x,y)

//Q4 : Changement de Variable attendu
r = 1:100
[e,f] = reglin(resultat(2,:), r)

/*CORRECTION*/
C0 = 13
R0 = 10
V0 = [C0,R0]
n = 50
A = [0.5, 0.4; -0.104, 1.1]
p = zeros(2,2)
p(1,1) = C0
p(2,1) = R0

for i = 2:n
    p(1:2, i) = a * p(1:2, i-1)
end

plot(p(1,:), 'bp')

//etc

//TP3
//1
x=poly(0,"x"); 
p=3+2*x^2+x^3;

X=[1:100];
plot(horner(p,X));

//2
Fichier=mopen('C:\Users\khali\Documents\polynome_bruit.txt','rt');
my_mat=mfscanf(-1,Fichier,'%f \t %f');
mclose(Fichier);

plot(my_mat(1,:),my_mat(2,:));

//3
br=log(my_mat)
plot(br(:,1),br(:,2),"blue");
plot(br(:,1),3*br(:,1),"green"); // le plus proche 
plot(br(:,1),4*br(:,1),"red");


//4

function cf = polyfit(x,y,n)
A = ones(length(x),n+1)
for i=1:n
    A(:,i+1) = x(:).^i
end
cf = lsq(A,y(:))
endfunction

x=my_mat(:,1);
y=my_mat(:,2);

//coef 2
n=2;
cf = polyfit(x,y,n);
t = linspace(min(x),max(x))' // now use these coefficients to plot the polynomial
A = ones(length(t),n+1)
for i=1:n
    A(:,i+1) = t.^i
end

subplot(3,2,1)
plot(x,y,'r')
subplot(3,2,2)
plot(t,A*cf,'b')

// coef 3
n=3;
cf = polyfit(x,y,n)

t = linspace(min(x),max(x))'   // now use these coefficients to plot the polynomial
A = ones(length(t),n+1)
for i=1:n
    A(:,i+1) = t.^i
end
subplot(3,2,3)
plot(x,y,'r')
subplot(3,2,4)
plot(t,A*cf)

//coef 4
n=4;
cf = polyfit(x,y,n)

t = linspace(min(x),max(x))'   // now use these coefficients to plot the polynomial
A = ones(length(t),n+1)
for i=1:n
    A(:,i+1) = t.^i
end
subplot(3,2,5)
plot(x,y,'r')
subplot(3,2,6)
plot(t,A*cf)

//residus
s=poly(0,'s');
H=[s/(s+1)^2,1/(s+2)];
N=numer(H);
D=denom(H);
w=residu(N.*horner(N,-s),D,horner(D,-s));  //N(s) N(-s) / D(s) D(-s)
plot(1:50,w)


// part 5: 
//1

Fichier=mopen('C:\Users\khali\Documents\kangourou.txt','rt');
Kc=mfscanf(-1,Fichier,'%f \t %f');
mclose(Fichier);
Kc=Kc';

//2
[H]=gsort(Kc,'lr' );
[a, b] = reglin(H(:,1)', H(:,2)');
X=[0:900];
plot(X,a*X+b);

// 3 la matrice C de covariance associ´ee aux donn´ees.
C = cov(Kc');

// 4
// D: matrice diagonale (valeurs propres)
// P: vecteurs propres
[P, D]=spec(C);

//5
Kvp=P * Kc;

//6

plot(Kvp(1,:),Kvp(2,:),'x');


[H]=gsort(Kvp,'lr' );
[c, d] = reglin(H(1,:), H(2,:));
plot(X,c*X+d);



// approx vue en cours
A=[Kvp(1,:),ones(4:1)];
M=A' * A;
B= A' * Kvp(2,:);
C=M\B;
t=0:0.1:10;
plot(t,C(1)*t+C(2),Kvp(1,:),Kvp(2,:),'x');


// Annales 
function [B]=miroir(a)
    n=size(a,2);
    for i= 1:n
        B(1,i)=a(i);
    end
    for i= 2:n
        tmp=B(i-1,n)
        for j= n:-1:2
            B(i,j)=B(i-1,j-1);
        end
        B(i,1)=tmp;
    end
endfunction

// Anales: projection
function [x]=produit_scalaire(A,j,B,k)
    n=size(A,1);
    sum=0;
    for i=1:n
        sum=sum+A(i,j)*B(i,k);
    end
    x=sum;
endfunction

function [x]=normalisation_vecteur(B,k)
    x=sqrt(produit_scalaire(B,k,B,k));
endfunction

function [a]=mult_scalaire_vecteur(c,A,k)
    n=size(A,1);
    for i=1:n
        y(i)=c*A(i,k);
    end
    a=y;
endfunction

function [a]=projection(E,j,A,k)
    sum=produit_scalaire(E,j,A,k);
    a=mult_scalaire_vecteur(sum,E,j);
endfunction

function [M]=sub_vecteur(A,k,B,j)
    n=size(A,1);
    for i=1:n
        M(i,1)=A(i,k)-B(i,j);
    end
endfunction


function [E]=gram_schmidt(A)
    B(:,1)=A(:,1);
    E(:,1)=B(:,1) / normalisation_vecteur(B,1);
    n=size(A,2);
    for i=2:n
        B(:,i)=A(:,i);
        for j=1:i-1
            B(:,i)=sub_vecteur(B,i,projection(E,j,A,i),1);
        end
        E(:,i)=B(:,i) / normalisation_vecteur(B,i);
    end
endfunction


// moidre carrés
function [x]=det_2x2(A)
    x=A(1,1)*A(2,2) - A(1,2)*A(2,1);
endfunction

function [Inv]=inverse_2x2(A)
    in=1/det_2x2(A);
    Inv(1,1)=in*A(2,2);
    Inv(1,2)=-in*A(1,2);
    Inv(2,1)=-in*A(2,1);
    Inv(2,2)=in*A(1,1);
endfunction

function [x,y]=resout(A,b)
    x=A(1,1)*b(1,1) + A(1,2)*b(2,1);
    y=A(2,1)*b(1,1) + A(2,2)*b(2,1);
endfunction

// pivot de gauss remontee

function [A,B]=addmultiple(A,B,i,j,c)
    n=size(A,2);
    for k=1:n
        A(i,k)=A(i,k)+ c*A(j,k);
    end
    B(i,1)=B(i,1)+c*B(j,1);
endfunction

function[X]=remontee(A,B)
    n=size(A,1);
    for i=n:-1:1
        X(i,1)=B(i,1);
        for j=i+1:n
            X(i,1)=X(i,1)-A(i,j)*X(j,1);
        end
        X(i,1)=X(i,1)/A(i,i);
    end
endfunction

function [r]=choixPivot(A,i)
    n=size(A,1);
    r=-1;
    for j=i:n
        if A(i,j) <> 0
            r=j;
        end
    end
endfunction

function [A,B]=echangeLigne(A,B,i,j)
    n=size(A,2);
    for k=1:n
        tmp=A(i,k);
        A(i,k)=A(j,k);
        A(j,k)=tmp;
    end
    tmp=B(i,1);
    B(i,1)=B(j,1);
    B(j,1)=tmp;
endfunction


function [A,B]=triangulaire_sup(A,B)
    n=size(A,1);
    for i=1:n-1
        [j]=choixPivot(A,i);
        [A,B]=echangeLigne(A,B,i,j);
        for j=i+1:n
            [A,B]=addmultiple(A,B,j,i,-A(j,i)/A(i,i));
        end
    end
endfunction

function [A,B,X]=gauss(A,B)
    [A,B]=triangulaire_sup(A,B);
    [X]=remontee(A,B);
endfunction

//
function [M]=chouette_rats(C,R,n)
    M(1,1)=C;
    M(2,1)=R;
    for i=2:n
        M(1,i)=0.5 * M(1,i-1) + 0.4 * M(2,i-1);
        M(2,i)=-0.104 * M(1,i-1) + 1.1 * M(2,i-1);
    end
endfunction




// representation des donnees
Fichier=mopen('C:\Users\khali\Documents\21815756.CSV','rt');
M=mfscanf(-1,Fichier,'%f \t %f');
mclose(Fichier);
subplot(3,1,2);
plot(M(1,:),M(2,:),"y");

// estimation par logarithme
br=log(M);
subplot(3,1,3);
plot(br(:,1),br(:,2),"blue");
plot(br(:,1),3*br(:,1),"green"); // le plus proche 
plot(br(:,1),4*br(:,1),"red");
plot(br(:,1),10*br(:,1),"y");
plot(br(:,1),1000*br(:,1),"black");
