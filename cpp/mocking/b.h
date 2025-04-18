#ifndef B_H
#define B_H

#include "ia.hpp"
#include "a.h"

class B {
    IA* a = nullptr;
public:
    explicit B(IA* ia = new A());
    ~B();
	
    int goo();
};

#endif // B_H