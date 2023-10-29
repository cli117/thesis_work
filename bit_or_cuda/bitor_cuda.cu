#include "stdio.h"


// Here we set the maximum length of pattern_length is 32
__global__ void build_mask_table(char* pattern, int pattern_length, unsigned long long* mask_table)
{
    int char_idx = blockIdx.x;
    int pat_idx = threadIdx.x;

    atomicAnd(&mask_table[char_idx], ~((int(pattern[pat_idx]) == char_idx) << (pattern_length - pat_idx - 1)));
    return;
}

__global__ void shift_or(char** string_cuda, int* packets_size_config, unsigned long long* mask_table, int pat_len, bool* filtered_valid)
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
            filtered_valid[blockIdx.x] = true;
        }
        __syncthreads();
        curr_pos += stride;
    }
}


__global__ void shift_or_optimized(char** targets_cuda, int* targets_size_config, unsigned long long* mask_table, int pat_len, bool* filtered_valid)
{
    char* target = targets_cuda[blockIdx.x];
    int m = targets_size_config[blockIdx.x];
    int stride = blockDim.x;
    unsigned long long curr_mask[5];
    int mask_len = 3 * pat_len - 1;
    
    

    for (int index = threadIdx.x; index < m; index += stride)
    {
        curr_mask[0] = 0;
        curr_mask[1] = 0;
        curr_mask[2] = 0;
        curr_mask[3] = 0;
        curr_mask[4] = 0;

        int i = pat_len * index;
        int j = pat_len * (index + 2)-1;
        if(i>m)
            return;
        if(j>m)
            j=m;

        while (i <= j)
        {
            curr_mask[0] <<= 1;
            curr_mask[0] |= curr_mask[1] >> 63;

            curr_mask[1] <<= 1;
            curr_mask[1] |= curr_mask[2] >> 63;

            curr_mask[2] <<= 1;
            curr_mask[2] |= curr_mask[3] >> 63;

            curr_mask[3] <<= 1;
            curr_mask[3] |= curr_mask[4] >> 63;

            curr_mask[4] <<= 1;
            curr_mask[4] |= mask_table[target[j]];

            j -= 1;
        }

        for (int k = 0; k < mask_len; k++)
        {
            if (!(curr_mask[k/64] & (1ULL << (k%64))))
            {
                filtered_valid[threadIdx.x] = true;
            }
        }
    }
}