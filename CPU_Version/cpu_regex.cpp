#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <chrono>
#include <iostream>

using namespace std;

struct cuda_pair
{
    int src;
    int dest;
};

void print_nfa(vector<cuda_pair>* nfa)
{
    for (int i = 0; i < 256; i++)
    {
        for (int j = 0; j < nfa[i].size(); j++)
        {
            cuda_pair transition = nfa[i][j];
            printf("%d - %d: %c\n", transition.src, transition.dest, i);
        }
    }
}

vector<string> get_packets()
{
    vector<string> ret;
    ifstream infile("strings.txt");
    string line;
    while (getline(infile, line))
    {
        if (line.length() > 0) {
            ret.push_back(line);
        }
    }
    
    return ret;
}

vector<cuda_pair>* get_nfa()
{
    vector<cuda_pair>* ret = (vector<cuda_pair>*) calloc(256, sizeof(vector<cuda_pair>));
    ifstream infile("patterns.nfa");
    string line;
    while (getline(infile, line))
    {
        size_t dash_idx = line.find_first_of("-");
        size_t cln_idx = line.find_first_of(":");

        int src = stoi(line.substr(0, dash_idx));
        int dest = stoi(line.substr(dash_idx+1, cln_idx - dash_idx - 1));
        if (src == dest){
            continue;
        }
        cuda_pair transition;
        transition.src = src;
        transition.dest = dest;
        char idx = line[cln_idx + 1];
        if (idx == '\\') {
            string context = line.substr(cln_idx+2, line.length() - cln_idx - 1);
            if (context.compare("d") == 0) {
                for (int i = 0; i < 10; i++) {
                    ret[i + '0'].push_back(transition);
                }
            } else if (context.compare("s") == 0)
            {
                ret[' '].push_back(transition);
            }
            
        } else {
            ret[line[cln_idx + 1]].push_back(transition);
        }
    }
    return ret;
}

int get_num_of_states() 
{
    int ret = 0;
    ifstream infile("patterns.nfa");
    string line;
    while (getline(infile, line))
    {
        size_t dash_idx = line.find_first_of("-");
        size_t cln_idx = line.find_first_of(":");
        int dest = stoi(line.substr(dash_idx+1, cln_idx - dash_idx - 1));
        ret = max(dest, ret);
    }

    return ret;
}

int* get_persistent_sv()
{
    int size = get_num_of_states();
    int* ret = (int*) calloc(size, sizeof(int));
    ifstream infile("patterns.nfa");
    string line;
    while (getline(infile, line))
    {
        size_t dash_idx = line.find_first_of("-");
        size_t cln_idx = line.find_first_of(":");

        int src = stoi(line.substr(0, dash_idx));
        int dest = stoi(line.substr(dash_idx+1, cln_idx - dash_idx - 1));
        if (src == dest){
            ret[src - 1] = 1;
        }
    }

    return ret;
}

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

int main() {
    vector<string> packets = get_packets();
    vector<cuda_pair>* nfa = get_nfa();
    int num_of_states = get_num_of_states();
    int* persistent_sv = get_persistent_sv();
    int* ret_vec = (int*) calloc(packets.size(), sizeof(int));

    auto start = chrono::steady_clock::now();
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

    cout << "Elapsed time in milliseconds: "
        << chrono::duration_cast<chrono::milliseconds>(end - start).count()
        << " ms" << endl;

    return 0;
}