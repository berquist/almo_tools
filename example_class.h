#ifndef EXAMPLE_CLASS_H_
#define EXAMPLE_CLASS_H_

#include "indices.h"

using namespace libresponse;

class example_class {

private:
    type::indices indices_ao;

public:
    example_class();
    ~example_class();

    void init();
};

#endif // EXAMPLE_CLASS_H_
