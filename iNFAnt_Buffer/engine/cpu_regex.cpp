#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>
#include <iostream>
#include "../utils/loader.h"

using namespace std;

void bit_vector_and(int* a, int* b, int* output, int size) {
    for (int i = 0; i < size; i++) {
        output[i] = a[i] & b[i];
    }
}

void vector_copy(int* src, int* dest, int size)
{
    for (int i = 0; i < size; i++)
    {
        dest[i] = src[i];
    }
}

int main(int argc, char *argv[]) {
    std::string string_filename = "strings";
    if (argc > 1) {
        string_filename = argv[1];
    }
    auto start = chrono::steady_clock::now();
    vector<string> packets = get_packets(string_filename);
    std::unordered_set<int> acc_set;
    std::vector<cuda_pair>* nfa = get_nfa("../test_suite/nfa_output/nfa0.nfa", &acc_set);
    int num_of_states = get_num_of_states();
    int* persistent_sv = get_persistent_sv();
    int* ret_vec = (int*) calloc(packets.size(), sizeof(int));

    
    for (int i = 0; i < packets.size(); i++) {
        int* c_vec = (int*)calloc(num_of_states, sizeof(int));
        int* f_vec = (int*)calloc(num_of_states, sizeof(int));
        string packet = packets[i];
        for (int j = 0; j < packets[i].length(); j++) {
            bit_vector_and(c_vec, persistent_sv, f_vec, num_of_states);
            vector<cuda_pair> transitions = nfa[packet[j]];
            for (int k = 0; k < transitions.size(); k++) {
                cuda_pair curr_transition = transitions[k];
                int src = curr_transition.src - 1;
                int dest = curr_transition.dest - 1;

                if (src == -1 || c_vec[src] == 1)
                {
                    f_vec[dest] = 1;
                }
            }

            vector_copy(f_vec, c_vec, num_of_states);
            if (c_vec[num_of_states - 1] == 1) {
                ret_vec[i] = 1;
            }
        }
    }
    auto end = chrono::steady_clock::now();
    
    // printf("========================================\n");
    // for (int i = 0; i < packets.size(); i++)
    // {
    //     if (ret_vec[i] == 1) {
    //         printf("%d: %s\n", i, packets[i].c_str());
    //     }
        
    // }
    // printf("\n========================================\n");

    cout << "CPU regex elapsed time in milliseconds: "
        << chrono::duration_cast<chrono::milliseconds>(end - start).count()
        << " ms" << endl;

    return 0;
}