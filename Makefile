test:
	g++ -larmadillo -I. -o indexing_test.x utils.C indices.C printing.C indexing_test.C

all:
	g++ -larmadillo -I. -o restricted_multiplication.x utils.C indices.C printing.C restricted_multiplication_main.C

clean:
	rm -f *.x
