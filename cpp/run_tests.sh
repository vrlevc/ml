#!/bin/bash

g++ -std=c++17 -I.. -o test_b test_b.cpp b.cpp a.cpp

if [ $? -ne 0 ]; then
    echo "Build failed."
    exit 1
fi

# Run the test
./test_b

if [ $? -ne 0 ]; then
    echo "Test failed."
    exit 1
else
    echo "All tests passed."
fi
