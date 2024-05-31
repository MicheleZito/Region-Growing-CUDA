#!/bin/bash

rm CMakeCache.txt
rm cmake_install.cmake
rm Makefile
rm -r CMakeFiles
cmake .
make
