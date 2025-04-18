#include "b.h"
#include <iostream>
#include <cassert>

// Mock class for IAbstractA
class MockA : public IA {
public:
    int foo() override {
        return 42; // Mocked behavior
    }
};

int main() {
    // Test case for B::goo
    MockA* pA = new MockA();
    B b(pA);

    int result = b.goo();
    assert(result == 210); // 42 * 5 = 210

    std::cout << "Test passed: B::goo() returned " << result << std::endl;
    return 0;
}
