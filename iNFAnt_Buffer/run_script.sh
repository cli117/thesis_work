cd engine
nvcc $1.cu -o $1 -g -G
if [ "$4" ]; then
    ./$1 $2 $3 $4
else
    ./$1 $2 $3
fi

# if [ "$2" ]; then
#     g++ $2.cpp -o $2
#     ./$2 $3
# fi