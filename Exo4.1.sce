////////////////////////////////
//Question 1
/////////////////////////////////////////////////


x   = poly(0,"x")
P   = x*3 + 2 * x*x + 3
//---
//REMARQUE : attention ici, petite faute (de frappe je pense) : il fallait écrire x^3 et non x*3. notons au passage qu'on aurait pu remplacer x*x par x^2.
//---

abs = 0:100
ord = horner(P,abs)
scf(1)
clf(1)
plot(abs,ord);
plot([0:100],horner(P,[0:100]))

////////////////////////////////
//Question 2
/////////////////////////////////////////////////

P = fscanfMat("polynome.txt");
size(P)

scf(2)
clf(2)

plot(P)

scf(3)
clf(3)

plot(P(:,1),P(:,2))
//colonne 1 et colonne 2

////////////////////////////////
//Question 3
/////////////////////////////////////////////////

x = P(:,1);
y = P(:,2);
xlog = log(x);
ylog = log(y)


scf(4)
clf(4)

plot(xlog,ylog)

[a,b]reglin(log(x)',log(y)')

disp(a)

// on retoruve a = 3 donc on aura un polynome de deg 3

////////////////////////////////
//Question 4
/////////////////////////////////////////////////

X2 = [x.^2, x ,ones(x)] //colonne
A2 = X2 \ y //donne les coeff du polynome de degré 2

z = poly(0,"z")
P2 = A2(1) * z *2 + A2(2) * z + A2(3)
//Pour savoir si c est une bonne regression lineaire on calcul les residus

// residu = y - P2(x)
// Calcul de residu 2 :

Residu2 = y - horner(P2,x)
//	res2    = sum(abs(Residu2))

// Avec un polynome 3

X3 = [x.^3,x.^2,x,ones(x)]
A3 = X3 \ y
P3 = [ z*3, z*2, z,1]*A3

//---
//REMARQUE : Je suppose que vous avez voule écrire *A3  plutot que **3 ? 
//---

