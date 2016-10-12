all:
	g++ -larmadillo -o restricted_multiplication.x restricted_multiplication.C

clean:
	rm -f *.x
