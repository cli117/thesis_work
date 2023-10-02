#include "../utils/loader.h"
#include "../../KMP/kmp_CUDA.cu"
#include "../../ASyncAP/ASyncAP.cu"
#include <chrono>
#include <filesystem>
namespace fs = std::filesystem;


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

__global__ void iNFAnt_match(char** packets_cuda, int* packets_size_config, cuda_pair** nfa_cuda, int* state_transition_size_cfg, int num_of_states, int* persistent_sv, bool* filtered_valid, int* acc_states, int acc_length)
{
    if (!filtered_valid[blockIdx.x])
    {
        return;
    }

    __shared__ int curr_pos;
    __shared__ int found;
    extern __shared__ int vector[];
    int* c_vector = &vector[0];
    int* f_vector = &vector[num_of_states];
    char* packet = packets_cuda[blockIdx.x];
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
        }

        for (int i = threadIdx.x; i < acc_length; i += stride)
        {
            if (c_vector[acc_states[i] - 1] == 1)
            {
                found = 1;
                printf("found!\n");
                break;
            }
        }

        __syncthreads();

        // early return if found
        if (found == 1) {
            return;
        }
    }
}

int main(int argc, char *argv[])
{   
    std::string string_filename = "strings";
    std::string working_dir = "";
    if (argc > 1) {
        working_dir = argv[1];
    }
    
    if (argc > 2)
    {
        string_filename = argv[2];
    }

    int iterations = -1;
    if (argc > 3)
    {
        iterations = stoi(argv[3]);
    }

    cuda_pair** nfa_cuda;
    int* state_transition_size_cfg;
    cudaMallocManaged(&nfa_cuda, 256*sizeof(cuda_pair*));
    cudaMallocManaged(&state_transition_size_cfg, 256*sizeof(int));

    // packets loading and allocation
    std::vector<std::string> packets = get_packets(working_dir + string_filename);
    char** packets_cuda;
    int* packets_size_config;
    int num_of_packets = packets.size();
    cudaMallocManaged(&packets_cuda, num_of_packets * sizeof(char*));
    cudaMallocManaged(&packets_size_config, num_of_packets * sizeof(int));

    for (int i = 0; i < num_of_packets; i++)
    {
        int str_len = packets[i].size();
        cudaMallocManaged(&packets_cuda[i], str_len * sizeof(char));
        strcpy(packets_cuda[i], packets[i].c_str());
        packets_size_config[i] = str_len;
    }

    auto start = std::chrono::steady_clock::now();
    // for (const auto & entry : fs::directory_iterator(working_dir + "test_suite/nfa_output"))
    for (int i = 0; i < 1; i++)
    {
        // std::string regex_file = entry.path();
        std::string regex_file = working_dir + "test_suite/nfa_output/nfa488.nfa";
        std::string corpus_file = working_dir + string_filename;

        std::unordered_set<int> acc_set;
        std::vector<cuda_pair>* nfa = get_nfa(regex_file, &acc_set);
        std::cout << "Searching " << regex_file << std::endl;

    
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
        
    // KMP Testing
        int cSize = 4;
        char test_pat[] = "qcp";
        int n = strlen(test_pat);
        int *f;
        f = new int[n];
        preKMP(test_pat, f);
        int *d_f;
        char *d_pat;
        cudaMalloc((void **)&d_f, n*cSize);
        cudaMalloc((void **)&d_pat, n*cSize);
        cudaMemcpy(d_pat, test_pat, n*cSize, cudaMemcpyHostToDevice);
        cudaMemcpy(d_f, f, n*cSize, cudaMemcpyHostToDevice);

        bool* filtered_valid;
        cudaMallocManaged(&filtered_valid, num_of_packets * sizeof(bool));

        KMP<<<num_of_packets, NUM_OF_THREADS>>>(d_pat, packets_cuda, packets_size_config, d_f, strlen(test_pat), num_of_packets, filtered_valid);
        cudaDeviceSynchronize();

        int num_of_states = get_num_of_states();
        int* persistent_sv = get_persistent_sv();

        int* persistent_sv_cuda;
        cudaMallocManaged(&persistent_sv_cuda, num_of_states*sizeof(int));

        for (int i = 0; i < num_of_states; i++)
        {
            persistent_sv_cuda[i] = persistent_sv[i];
        }

        int* acc_states;
        cudaMallocManaged(&acc_states, acc_set.size() * sizeof(int));
        int cnt = 0;
        for (int state: acc_set)
        {
            acc_states[cnt] = state;
            cnt += 1;
        }

        iNFAnt_match<<<num_of_packets, NUM_OF_THREADS, 2*num_of_states*sizeof(int)>>>(packets_cuda, packets_size_config, nfa_cuda, state_transition_size_cfg, num_of_states, persistent_sv_cuda, filtered_valid, acc_states, acc_set.size());
    
// ASyncAPTesting
        // ASyncAP<<<num_of_packets, NUM_OF_THREADS, 2*num_of_states*sizeof(int)>>>(packets_cuda, packets_size_config, nfa_cuda, state_transition_size_cfg, num_of_states, persistent_sv_cuda, acc_states, acc_set.size());
    
    
        for (int i = 0; i < 256; i++)
        {
            cudaFree(nfa_cuda[i]);
        }
        cudaFree(filtered_valid);
        cudaFree(persistent_sv_cuda);
        cudaFree(acc_states);
    }


    cudaDeviceSynchronize();
    // printf("========================================\n");
    // for (int i = 0; i < num_of_packets; i++)
    // {
    //     if (ret_vec[i] == 1) {
    //         printf("%d: %s\n", i, packets[i].c_str());
    //     }
        
    // }
    // printf("\n========================================\n");
    
    // free allocated memory
    
    cudaFree(nfa_cuda);
    cudaFree(state_transition_size_cfg);
    for (int i = 0; i < num_of_packets; i++)
    {
        cudaFree(packets_cuda[i]);
    }

    cudaFree(packets_cuda);
    cudaFree(packets_size_config);

    auto end = std::chrono::steady_clock::now();
    std::cout << "GPU elapsed time in milliseconds" << "(num of threads: " << NUM_OF_THREADS << "): "
        << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
        << " ms" << std::endl;
    return 0;
}