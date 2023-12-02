__device__ void vector_copy_AP(int* src, int* dest, int size, int start_pos)
{
    for (int i = start_pos; i < start_pos + size; i++)
    {
        dest[i] = src[i];
    }
}

__device__ void print_vec_AP(int* vec, int size)
{
    if (threadIdx.x == 0)
    {
        for (int i = 0; i < size; i++)
        {
            printf("%d ", vec[i]);
        }

        printf("%c", '\n');
    }
}

__device__ void bit_vector_and_AP(int* a, int* b, int* output, int size, int start_pos)
{
    for (int i = start_pos; i < start_pos + size; i++) {
        output[i] = a[i] & b[i];
    }
}

__global__ void ASyncAP(char** packets_cuda, int* packets_size_config, cuda_pair** nfa_cuda, int* state_transition_size_cfg, int num_of_states, int* persistent_sv, int* acc_states, int acc_length, char* regex_filename, bool* result, bool* filtered_valid, int start_state = -1)
{
    // if (!filtered_valid[blockIdx.x])
    // {
    //     return;
    // }
    char* packet = packets_cuda[blockIdx.x];
    int n = packets_size_config[blockIdx.x];
    int stride = blockDim.x;
    __shared__ int c_vector[4096];
    __shared__ int f_vector[4096];

    int start_pos = threadIdx.x * 8;
    int size = num_of_states/32 + 1;

    for (int i = start_pos; i < start_pos + size; i++) 
    {
        c_vector[i] = 0;
        f_vector[i] = 0;
    }

    int curr_start_pos = threadIdx.x;
    while (curr_start_pos < n)
    {
        if (result[0])
        {
            return;
        }

        int curr_pos = curr_start_pos;

        while (curr_pos < n)
        {
            bit_vector_and_AP(c_vector, persistent_sv, f_vector, size, start_pos);
            int num_of_transitions = state_transition_size_cfg[packet[curr_pos]];
            bool proceed = false;
            for (int i = 0; i < num_of_transitions; i += 1)
            {
                cuda_pair curr_transition = nfa_cuda[packet[curr_pos]][i];
                int src = curr_transition.src - 1;
                int dest = curr_transition.dest - 1;
                
                if (src == start_state || (c_vector[start_pos + (src / 32)] & (1 << (src % 32))))
                {
                    // printf("%c: %d - %d in %s\n", packet[curr_pos], src, dest, regex_filename);
                    f_vector[start_pos + (dest / 32)] |= 1 << (dest % 32);
                    proceed = true;
                    for (int i = threadIdx.x; i < acc_length; i += stride)
                    {
                        int offset = (acc_states[i] - 1) / 32;
                        if (f_vector[start_pos + offset] & (1 << ((acc_states[i] - 1) % 32)))
                        {
                            result[0] = true;
                            printf("found at %s!\n", regex_filename);
                            return;
                        }
                    }
                }
            }

            if (!proceed)
            {
                for (int i = start_pos; i < start_pos + size; i++) 
                {
                    c_vector[i] = 0;
                    f_vector[i] = 0;
                }
                break;
            }

            vector_copy_AP(f_vector, c_vector, size, start_pos);
            curr_pos += 1;
        }

        curr_start_pos += stride;
    }
}