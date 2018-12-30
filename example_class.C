#include "example_class.h"

example_class::example_class()
{ }

example_class::~example_class()
{ }

void example_class::init()
{
    const size_t dim = 4;
    arma::uvec nbasis_frgm(dim);
    for (size_t i = 0; i < dim; i++)
        nbasis_frgm(i) = i + 1;
    nbasis_frgm.print("nbasis_frgm");

    indices_ao = make_indices_ao(nbasis_frgm);
}
