#ifndef PRINTING_H_
#define PRINTING_H_

/*!
 * @brief Enable printing to streams for all typedefs.
 *
 * @file
 */

#include "typedefs.h"

std::ostream& operator<<(std::ostream& os, const indices& i);
std::ostream& operator<<(std::ostream& os, const pair_indices& pi);
std::ostream& operator<<(std::ostream& os, const pair_arma& pa);
std::ostream& operator<<(std::ostream& os, const pair& p);
std::ostream& operator<<(std::ostream& os, const pairs& ps);
// template <typename T>
// std::ostream& operator<<(std::ostream& os, const std::vector<T>& v);
template <typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& v) {

    typename std::vector<T>::const_iterator it;

    os << "[";
    for (it = v.begin(); it != v.end(); ++it) {
        if (it == v.end() - 1)
            os << *it;
        else
            os << *it << ", ";
    }
    os << "]" << std::endl;

    return os;

};

#endif // PRINTING_H_
