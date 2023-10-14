__device__ void vector_copy_AP(int* src, int* dest, int size)
{
    for (int i = 0; i < size; i++)
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

__device__ void bit_vector_and_AP(int* a, int* b, int* output, int size)
{
    for (int i = 0; i < size; i++) {
        output[i] = a[i] & b[i];
    }
}

__global__ void ASyncAP(char** packets_cuda, int* packets_size_config, cuda_pair** nfa_cuda, int* state_transition_size_cfg, int num_of_states, int* persistent_sv, int* acc_states, int acc_length, char* regex_filename, bool* result, bool* filtered_valid)
{
    if (!filtered_valid[blockIdx.x])
    {
        return;
    }
    char* packet = packets_cuda[blockIdx.x];
    int n = packets_size_config[blockIdx.x];
    int stride = blockDim.x;
    int c_vector[1000];
    int f_vector[1000];

    for (int i = 0; i < num_of_states; i++) 
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
            bit_vector_and_AP(c_vector, persistent_sv, f_vector, num_of_states);
            int num_of_transitions = state_transition_size_cfg[packet[curr_pos]];
            bool proceed = false;
            for (int i = 0; i < num_of_transitions; i += 1)
            {
                cuda_pair curr_transition = nfa_cuda[packet[curr_pos]][i];
                int src = curr_transition.src - 1;
                int dest = curr_transition.dest - 1;
                
                if (src == -1 || c_vector[src] == 1)
                {
                    // printf("%c: %d - %d\n", packet[curr_pos], src, dest);
                    f_vector[dest] = 1;
                    proceed = true;
                    for (int i = threadIdx.x; i < acc_length; i += stride)
                    {
                        if (f_vector[acc_states[i] - 1] == 1)
                        {
                            result[0] = true;
                            // printf("found at %s!\n", regex_filename);
                            return;
                        }
                    }
                }
            }

            if (!proceed)
            {
                for (int i = 0; i < num_of_states; i++) 
                {
                    c_vector[i] = 0;
                    f_vector[i] = 0;
                }
                break;
            }

            vector_copy_AP(f_vector, c_vector, num_of_states);
            curr_pos += 1;

            if (c_vector[num_of_states - 1] == 1) 
            {
                result[0] = true;
                return;
            }
        }

        curr_start_pos += stride;
    }
}