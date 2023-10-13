#include "stdio.h"


// Here we set the maximum length of pattern_length is 32
__global__ void build_mask_table(char* pattern, int pattern_length, unsigned long long* mask_table)
{
    int char_idx = blockIdx.x;
    int pat_idx = threadIdx.x;

    atomicAnd(&mask_table[char_idx], ~((int(pattern[pat_idx]) == char_idx) << (pattern_length - pat_idx - 1)));
    return;
}

__global__ void shift_or(char** string_cuda, int* packets_size_config, unsigned long long* mask_table, int pat_len)
{
    char* string = string_cuda[blockIdx.x];
    int str_len = packets_size_config[blockIdx.x];
    int stride = blockDim.x;
    __shared__ unsigned long long prev;
    __shared__ int found;

    if (threadIdx.x == 0)
    {
        found = 0;
        prev = ~0;
    }
    __syncthreads();

    int curr_pos = threadIdx.x;
    unsigned long long curr_mask;
    while (!found && curr_pos < str_len)
    {
        curr_mask = prev >> (65 - pat_len);
        __syncthreads();
        prev = 0;
        __syncthreads();
        curr_mask |= mask_table[string[curr_pos]] << threadIdx.x;
        atomicOr(&prev, curr_mask);
        __syncthreads();
        if (!(prev & (1ULL << threadIdx.x)))
        {
            atomicAdd(&found, 1);
            printf("found! at packet %d, %d\n", blockIdx.x, curr_pos);
        }
        __syncthreads();
        curr_pos += stride;
    }
}
