cd engine
nvcc $1.cu -o $1 -g -G --maxrregcount=64 -std=c++17 -Xcompiler "/std:c++17"

# if [ "$2" ]; then
#     g++ $2.cpp -o $2
#     ./$2 $3
# fi