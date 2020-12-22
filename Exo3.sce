//Question 1

n = 20;
C0 = 13;
R0 = 10;

A  = [0.5 0.4 ;-0.104 1.1];
V0 = [C0 ; R0];

V = zeros(2,2); //matrice qui contient que des zero
V(:,1) = V0; 

for i = 2:n
	V(:,i) = A * V(:,i-1);
end

scf(1)

plot(V(1,:),V(2,:),"-"); // ligne et ligne 2

//trace 
scf(2);
clf(2);

plot(V(1,:),'bp:');
plot(V(2,:),'gs-');

//trace 
scf(3)
clf(3)

plot(1:20,V(1,:),"+");
plot(1:20,V(2,:),"b");
//exec("Exo2.sce")

//(abs,ord,opt) opt:-couleur R/G/B -typepoint +,O,x,*  - style ligne - , --, -. 

////////////////////////////////
//Question 2
/////////////////////////////////////////////////

[A2,B2] = reglin(V(1,:),V(2,:)) // a2,b2:

[A2,B2] = reglin(V(1,10:20),V(2,10:20)) // on cherche les coord on les deux vont dans la meme direction

V(2,20)/V(1,20); //coeficient directeur des deux courbes

//correction/////////////////////////////////////
a = get("curent_axes");
a.x_location = "origin";
a.y_location = " origine";
plot(V(1,:),V(2,:),'bp:')
[A2,B2] = reglin(V(1,10:20),V(2,10:20)) 
/////////////////////////////////////////////////
//Question 3
////////////////////////////////
scf(4)
plot(V(2,1:19),V(2,2:20))

//pour afficher Ck en fonction de Ck-1

scf(5)
//plot2d(V(1,1:19,V(1,2:20),style = -1))
scf(6)
plot2d(V(1,1:n-1),V(1,2:n))
[A3,B3] = reglin(V(1,10:19),V(1,11:20)) 


/////////////////////////////////////////////////
//Question 4
////////////////////////////////

plot2d(0:(n-1),log(V(1,:)))
[A4,B4] = reglin(10:20,log(V(1,10:20))) 
//exprimer ck en fonction de t : ck= C0 * t puissance k
//log(ck)=log(C0) + k * log(T)
/////////////////////////////////////////////////
//Question 5
////////////////////////////////
scf(10)
clf(10)

subplot(221)
plot(V(1,:),'bp:')
plot(V(2,:),'gs-')

subplot(222)
plot(V(1,:),V(2,:),'bp:')

subplot(223)
plot2d(V(1,1:n-1),V(1,2:n),style=-1)

subplot(224)
plot2d(0:(n-1),log(V(1,:)),style=-1)
