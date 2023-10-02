#include "stdio.h"

__global__ void build_mask_table(char* pattern, int pattern_length, int** table)
{
    int char_idx = blockIdx.x;
    int pat_idx = threadIdx.x;
    if (char_idx > 255 || pat_idx >= pattern_length)
    {
        return;
    }
    
    table[char_idx][pat_idx] = int(pattern[pat_idx]) != char_idx;
    return;
}