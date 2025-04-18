#!/bin/bash

# Compile the source files into object files with include paths
g++ -I. -c a.cpp -o a.o
g++ -I. -c b.cpp -o b.o
g++ -I. -c main.cpp -o main.o

# Link the object files into an executable
g++ a.o b.o main.o -o application

# Clean up intermediate object files
rm a.o b.o main.o

echo "Build complete. Run ./application to execute the program."
