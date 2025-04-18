#include <iostream>

class A {
public:
    A() {
        std::cout << "A constructed" << std::endl;
    }
    ~A() {
        std::cout << "A destructed" << std::endl;
    }
};

int main() {
    std::cout << "Hello, World!" << std::endl;
    
    try {
        A a;
        throw std::runtime_error("An error occurred");
    } catch (const std::exception& e) {
        std::cout << "Caught exception: " << e.what() << std::endl;
    }
    
    return 0;
}