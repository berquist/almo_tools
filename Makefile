CXX = clang++

test:
	$(CXX) -larmadillo -I. -o indexing_test.x utils.C indices.C printing.C indexing_test.C

all:
	$(CXX) -larmadillo -I. -o restricted_multiplication.x utils.C indices.C printing.C restricted_multiplication_main.C

clean:
	rm -f *.x

pytest:
	pytest -v --doctest-modules --cov=almo_tools tests
