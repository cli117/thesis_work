__global__ void literal_match(char* pattern, char** targets_cuda, int* targets_size_config,int pat_len, int num_of_packets, bool* filtered_valid)
{
    char* target = targets_cuda[blockIdx.x];
    int m = targets_size_config[blockIdx.x];
    int stride = blockDim.x;

    int curr_start_pos = threadIdx.x;
    int ptr = 0;

    for (; curr_start_pos <= m - pat_len; curr_start_pos += stride)
    {
        filtered_valid[blockIdx.x] = false;
        ptr = 0;
        while (ptr < pat_len && target[curr_start_pos + ptr] == pattern[ptr])
        {
            ptr += 1;
        }

        if (ptr == pat_len)
        {
            filtered_valid[blockIdx.x] = true;
            return;
        }
    }
}