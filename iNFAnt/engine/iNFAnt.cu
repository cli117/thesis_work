#include "../utils/loader.h"
#include <chrono>

const int NUM_OF_THREADS = 5;

__device__ void print_vec(int* vec, int size)
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

__device__ void vector_copy(int* src, int* dest, int size)
{
    for (int i = 0; i < size; i++)
    {
        dest[i] = src[i];
    }
}

__device__ void bit_vector_and(int* a, int* b, int* output, int size)
{
    int stride = blockDim.x;
    for (int i = threadIdx.x; i < size; i += stride) {
        output[i] = a[i] & b[i];
    }
}

__global__ void iNFAnt_match(char* packets_cuda, int* packets_size_config, cuda_pair** nfa_cuda, int* state_transition_size_cfg, int num_of_states, int* persistent_sv, int* ret_vec)
{
    __shared__ int curr_pos;
    __shared__ int found;
    extern __shared__ int vector[];
    int* c_vector = &vector[0];
    int* f_vector = &vector[num_of_states];
    char* packet = packets_cuda + (blockIdx.x * 20000);
    int stride = blockDim.x;

    // initialize the bit vector for each block
    for (int i = threadIdx.x; i < num_of_states; i+= stride) {
        c_vector[i] = 0;
    }

    if (threadIdx.x == 0) {
        curr_pos = 0;
        found = 0;
    }
    
    __syncthreads();
    
    while (curr_pos < packets_size_config[blockIdx.x])
    {
        // printf("thread %d from block %d is processing: %c from packet %s with size %d\n", threadIdx.x, blockIdx.x, packet[curr_pos], packet, packets_size_config[blockIdx.x]);
        int num_of_transitions = state_transition_size_cfg[packet[curr_pos]];
        bit_vector_and(c_vector, persistent_sv, f_vector, num_of_states);
        // print_vec(c_vector, num_of_states);
        __syncthreads();
        for (int i = threadIdx.x; i < num_of_transitions; i += stride)
        {
            cuda_pair curr_transition = nfa_cuda[packet[curr_pos]][i];
            int src = curr_transition.src - 1;
            int dest = curr_transition.dest - 1;
            // printf("%d - %d \n", src, dest);
            if (src == -1 || c_vector[src] == 1)
            {
                f_vector[dest] = 1;
            }
            // print_vec(f_vector, num_of_states);
        }
        __syncthreads();
        // proceed to next character, execute once per block
        if (threadIdx.x == 0)
        {
            vector_copy(f_vector, c_vector, num_of_states);
            curr_pos += 1;
            if (c_vector[num_of_states - 1] == 1) {
                found = 1;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
    {
        ret_vec[blockIdx.x] = found;
    }
}

int main(int argc, char *argv[])
{   
    std::string string_filename = "strings";
    if (argc > 1) {
        string_filename = argv[1];
    }
    auto start = std::chrono::steady_clock::now();
    std::vector<cuda_pair>* nfa = get_nfa();
    // nfa loading and allocation
    int* state_transition_size_cfg;
    cuda_pair** nfa_cuda;
    cudaMallocManaged(&state_transition_size_cfg, 256*sizeof(int));
    cudaMallocManaged(&nfa_cuda, 256*sizeof(cuda_pair*));
    
    for (int i = 0; i < 256; i++)
    {
        int size = nfa[i].size();
        state_transition_size_cfg[i] = size;
        cudaMallocManaged(&nfa_cuda[i], size * sizeof(cuda_pair));
        for (int j = 0; j < size; j++)
        {
            nfa_cuda[i][j] = nfa[i][j];
        }
    }

    // packets loading and allocation
    std::string packets = get_packets(string_filename);
    int PACKET_UNIT_LENGTH = 20000;
    char* packets_cuda;
    int* packets_size_config;
    int num_of_packets = packets.length() / PACKET_UNIT_LENGTH + 1;
    cudaMallocManaged(&packets_cuda, packets.length() * sizeof(char));
    cudaMallocManaged(&packets_size_config, num_of_packets * sizeof(int));
    strcpy(packets_cuda, packets.c_str());
    for (int i = 0; i < num_of_packets; i++)
    {
        if (i < num_of_packets - 1) {
            packets_size_config[i] = PACKET_UNIT_LENGTH;
        } else {
            packets_size_config[i] = packets.length() - (PACKET_UNIT_LENGTH * (num_of_packets - 1));
        }
    }

    int num_of_states = get_num_of_states();
    int* persistent_sv = get_persistent_sv();

    int* persistent_sv_cuda;
    cudaMallocManaged(&persistent_sv_cuda, num_of_states*sizeof(int));

    for (int i = 0; i < num_of_states; i++)
    {
        persistent_sv_cuda[i] = persistent_sv[i];
    }

    int* ret_vec;
    cudaMallocManaged(&ret_vec, num_of_packets*sizeof(int));

    iNFAnt_match<<<num_of_packets, NUM_OF_THREADS, 2*num_of_states*sizeof(int)>>>(packets_cuda, packets_size_config, nfa_cuda, state_transition_size_cfg, num_of_states, persistent_sv_cuda, ret_vec);

    cudaDeviceSynchronize();
    printf("========================================\n");
    // for (int i = 0; i < num_of_packets; i++)
    // {
    //     if (ret_vec[i] == 1) {
    //         printf("found at index: %d", i);
    //     }
    // }
    printf("\n========================================\n");
    
    // free allocated memory
    for (int i = 0; i < 256; i++)
    {
        cudaFree(nfa_cuda[i]);
    }
    cudaFree(nfa_cuda);
    cudaFree(state_transition_size_cfg);
    cudaFree(packets_cuda);
    cudaFree(packets_size_config);

    auto end = std::chrono::steady_clock::now();
    std::cout << "GPU elapsed time in milliseconds" << "(num of threads: " << NUM_OF_THREADS << "): "
        << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
        << " ms" << std::endl;
    return 0;
}