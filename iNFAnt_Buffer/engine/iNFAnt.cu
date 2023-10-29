#include "../utils/loader.h"
#include "../../KMP/kmp_CUDA.cu"
#include "../../ASyncAP/ASyncAP.cu"
#include "../../bit_or_cuda/bitor_cuda.cu"
#include <chrono>
#include <filesystem>
#include <iostream>
#include <fstream>
namespace fs = std::filesystem;

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

__global__ void iNFAnt_match(char** packets_cuda, int* packets_size_config, cuda_pair** nfa_cuda, int* state_transition_size_cfg, int num_of_states, int* persistent_sv, bool* filtered_valid, int* acc_states, int acc_length, const char* regex_filename, bool* result)
{
    // if (!filtered_valid[blockIdx.x])
    // {
    //     return;
    // }
    __shared__ int curr_pos;
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
    }
    
    __syncthreads();
    
    while (!result[0] && curr_pos < packets_size_config[blockIdx.x])
    {
        // printf("thread %d from block %d is processing: %dth\n", threadIdx.x, blockIdx.x, curr_pos);
        int num_of_transitions = state_transition_size_cfg[packet[curr_pos]];
        bit_vector_and(c_vector, persistent_sv, f_vector, num_of_states);
        // print_vec(c_vector, num_of_states);
        __syncthreads();
        for (int i = threadIdx.x; i < num_of_transitions; i += stride)
        {
            cuda_pair curr_transition = nfa_cuda[packet[curr_pos]][i];
            int src = curr_transition.src - 1;
            int dest = curr_transition.dest - 1;
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
                result[0] = true;
                // printf("found at %s!\n", regex_filename);
                break;
            }
        }

        __syncthreads();

        // early return if found
        if (result[0]) {
            return;
        }
    }
}

int main(int argc, char *argv[])
{   
    std::string string_filename = "strings";
    std::string working_dir = "";
    std::string mode = "iNFAnt";
    std::string num_of_threads = "1024";
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

    if (argc > 4)
    {
        if (!strcmp(argv[4], "0") || !strcmp(argv[4], "infant")|| !strcmp(argv[4], "iNFAnt"))
        {
            mode = "iNFAnt";
        }
        else
        {
            mode = "ASyncAP";
        }
    }

    if (argc > 5)
    {
        num_of_threads = argv[5];
    }

    const int NUM_OF_THREADS = stoi(num_of_threads);
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
    std::cout << mode << " matching..." << std::endl;
    auto start = std::chrono::steady_clock::now();
    int num_of_iterations = 0;
    float time_prefilter = 0.;
    ofstream f;
    f.open("/home/cli117/Documents/regex.log");
    for (const auto & entry : fs::directory_iterator(working_dir + "test_suite/nfa_compare"))
    // for (int i = 0; i < 1; i++)
    {
        num_of_iterations += 1;
        if (iterations > 0 && num_of_iterations > iterations)
        {
            break;
        }

        std::string regex_file = entry.path().u8string();
        // std::string regex_file = working_dir + "test_suite/nfa_output/nfa65.nfa";
        f << num_of_iterations << " th iteration: " << regex_file << std::endl;
        std::cout << num_of_iterations << " th iteration: " << regex_file << std::endl;
        std::string corpus_file = working_dir + string_filename;

        std::unordered_set<int> acc_set;
        std::vector<cuda_pair>* nfa = get_nfa(regex_file, &acc_set);

    
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

        
        auto filter_start = std::chrono::steady_clock::now();
    // KMP Testing
        int cSize = 4;
        int pat_len = 0;
        string pat = get_longest_literal(regex_file, pat_len);
        char *d_pat;
        cudaMalloc((void **)&d_pat, pat_len*cSize);
        cudaMemcpy(d_pat, pat.c_str(), pat_len*cSize, cudaMemcpyHostToDevice);
        bool* filtered_valid;
        cudaMallocManaged(&filtered_valid, num_of_packets * sizeof(bool));


        // int *f;
        // f = new int[pat_len];
        // preKMP(pat.c_str(), f);
        // int *d_f;
        // cudaMalloc((void **)&d_f, pat_len*cSize);
        // cudaMemcpy(d_f, f, pat_len*cSize, cudaMemcpyHostToDevice);

        // KMP<<<num_of_packets, NUM_OF_THREADS>>>(d_pat, packets_cuda, packets_size_config, d_f, pat_len, num_of_packets, filtered_valid);



        // unsigned long long* mask_table;
        // cudaMallocManaged(&mask_table, 256*sizeof(unsigned long long));
        // for (int i = 0; i < 256; i++)
        // {
        //     mask_table[i] = ~0;
        //     mask_table[i] = mask_table[i] >> (64 - pat_len);
        // }
        // build_mask_table<<<256, pat_len>>>(d_pat, pat_len, mask_table);
        // int shift_or_threads = 65 - pat_len;
        // shift_or<<<num_of_packets, shift_or_threads>>>(packets_cuda, packets_size_config, mask_table, pat_len, filtered_valid);
        // shift_or_optimized<<<num_of_packets, NUM_OF_THREADS>>>(packets_cuda, packets_size_config, mask_table, pat_len, filtered_valid);
        
        cudaDeviceSynchronize();
        time_prefilter += std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - filter_start).count();

        int num_of_states = get_num_of_states(regex_file);
        int* persistent_sv = get_persistent_sv(regex_file);

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
        char* regex_file_cuda;
        cudaMallocManaged(&regex_file_cuda, sizeof(char) * regex_file.size());
        strcpy(regex_file_cuda, regex_file.c_str());
        bool* result;
        cudaMallocManaged(&result, sizeof(bool));
        result[0] = false;
        if (mode == "iNFAnt")
        {
            iNFAnt_match<<<num_of_packets, NUM_OF_THREADS, 2*num_of_states*sizeof(int)>>>(packets_cuda, packets_size_config, nfa_cuda, state_transition_size_cfg, num_of_states, persistent_sv_cuda, filtered_valid, acc_states, acc_set.size(), regex_file_cuda, result);
        }
        else if (mode == "ASyncAP")
        {
            ASyncAP<<<num_of_packets, NUM_OF_THREADS>>>(packets_cuda, packets_size_config, nfa_cuda, state_transition_size_cfg, num_of_states, persistent_sv_cuda, acc_states, acc_set.size(), regex_file_cuda, result, filtered_valid);
        }
        cudaDeviceSynchronize();

        for (int i = 0; i < 256; i++)
        {
            cudaFree(nfa_cuda[i]);
        }
        cudaFree(filtered_valid);
        cudaFree(persistent_sv_cuda);
        cudaFree(acc_states);
        cudaFree(d_pat);
    }

    f.close();

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

    std::cout << "Prefilter time consumption: " << time_prefilter << "ms" << std::endl;
    auto end = std::chrono::steady_clock::now();
    std::cout << "In mode " << mode << "GPU elapsed time in milliseconds" << "(num of threads: " << NUM_OF_THREADS << "): "
        << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
        << " ms" << std::endl;

    // char pat[] = "KFC Crazy Thursday! vivo 50";
    // int pat_len = strlen(pat);
    // unsigned long long* mask_table;
    // cudaMallocManaged(&mask_table, 256*sizeof(unsigned long long));
    // for (int i = 0; i < 256; i++)
    // {
    //     mask_table[i] = ~0;
    //     mask_table[i] = mask_table[i] >> (64 - pat_len);
    // }
    // char* pat_cuda;
    // cudaMallocManaged(&pat_cuda, sizeof(char) * pat_len);
    // strcpy(pat_cuda, pat);
    // build_mask_table<<<256, pat_len>>>(pat_cuda, pat_len, mask_table);
    // cudaDeviceSynchronize();
    // // for (int i = 0; i < 256; i++)
    // // {
    // //     std::cout << mask_table[i] << " " << std::endl;
    // // }

    // int shift_or_threads = 65 - pat_len;
    // shift_or<<<num_of_packets, shift_or_threads>>>(packets_cuda, packets_size_config, mask_table, pat_len);
    // cudaDeviceSynchronize();
    return 0;
}