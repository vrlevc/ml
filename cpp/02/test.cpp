#include <iostream>
#include <vector>
#include "main.cpp"

using namespace std;

void runTests() {
    Solution solution;

    vector<int> ratings1 = {1, 0, 2};
    cout << "Test 1: " << (solution.candy(ratings1) == 5 ? "Passed" : "Failed") << endl;

    vector<int> ratings2 = {1, 2, 2};
    cout << "Test 2: " << (solution.candy(ratings2) == 4 ? "Passed" : "Failed") << endl;

    vector<int> ratings3 = {1, 3, 2, 2, 1};
    cout << "Test 3: " << (solution.candy(ratings3) == 7 ? "Passed" : "Failed") << endl;
}

int main() {
    runTests();
    return 0;
}
