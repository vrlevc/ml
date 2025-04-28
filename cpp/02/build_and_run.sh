#!/bin/bash

# Build the main program and test file
g++ -o main main.cpp test.cpp

# Check if the build was successful
if [ $? -eq 0 ]; then
    echo "Build successful. Running the tests..."
    # Run the tests
    ./main
else
    echo "Build failed. Please check the errors above."
fi
