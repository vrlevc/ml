#!/bin/bash

# Build the application
g++ -o main main.cpp

# Check if the build was successful
if [ $? -eq 0 ]; then
    echo "Build successful. Running the application..."
    # Run the application
    ./main
else
    echo "Build failed. Please check the errors above."
fi
