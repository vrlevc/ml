#include "b.h"

B::B(IA* ia) : a(ia) {
}

B::~B() {
    delete a;
}

int B::goo() {
    return a->foo() * 5;
}