cd engine
nvcc $1.cu -o $1
if [ "$4" ]; then
    sudo nvprof --unified-memory-profiling off ./$1 $3
else
    ./$1 $3
fi

# if [ "$2" ]; then
#     g++ $2.cpp -o $2
#     ./$2 $3
# fi