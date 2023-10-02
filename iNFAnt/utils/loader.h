#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <iostream>

struct cuda_pair
{
    int src;
    int dest;
};

std::vector<cuda_pair>* get_nfa()
{
    std::vector<cuda_pair>* ret = (std::vector<cuda_pair>*) calloc(256, sizeof(std::vector<cuda_pair>));
    std::ifstream infile("../test_suite/patterns.nfa");
    std::string line;
    while (std::getline(infile, line))
    {
        size_t dash_idx = line.find_first_of("-");
        size_t cln_idx = line.find_first_of(":");

        int src = std::stoi(line.substr(0, dash_idx));
        int dest = std::stoi(line.substr(dash_idx+1, cln_idx - dash_idx - 1));
        if (src == dest){
            continue;
        }
        cuda_pair transition;
        transition.src = src;
        transition.dest = dest;
        char idx = line[cln_idx + 1];
        if (idx == '\\') {
            std::string context = line.substr(cln_idx+2, line.length() - cln_idx - 1);
            if (context.compare("d") == 0) {
                for (int i = 0; i < 10; i++) {
                    ret[i + '0'].push_back(transition);
                }
            } else if (context.compare("s") == 0)
            {
                ret[' '].push_back(transition);
            } else {
                std::string hex_str = context.substr(1, 2);
                char hex_char = std::stoi(hex_str, nullptr, 16);
                // std::cout << hex_char << std::endl;
                ret[hex_char].push_back(transition);
            }
            
        } else {
            if (idx == '.') {
                for (int i = 0; i < 256; i++) {
                    ret[i].push_back(transition);
                }
            } else {
                ret[idx].push_back(transition);
            }
        }
    }
    return ret;
}

int get_num_of_states() 
{
    int ret = 0;
    std::ifstream infile("../test_suite/patterns.nfa");
    std::string line;
    while (std::getline(infile, line))
    {
        size_t dash_idx = line.find_first_of("-");
        size_t cln_idx = line.find_first_of(":");
        int dest = std::stoi(line.substr(dash_idx+1, cln_idx - dash_idx - 1));
        ret = std::max(dest, ret);
    }

    return ret;
}

int* get_persistent_sv()
{
    int size = get_num_of_states();
    int* ret = (int*) calloc(size, sizeof(int));
    std::ifstream infile("../test_suite/patterns.nfa");
    std::string line;
    while (std::getline(infile, line))
    {
        size_t dash_idx = line.find_first_of("-");
        size_t cln_idx = line.find_first_of(":");

        int src = std::stoi(line.substr(0, dash_idx));
        int dest = std::stoi(line.substr(dash_idx+1, cln_idx - dash_idx - 1));
        if (src == dest){
            ret[src - 1] = 1;
        }
    }

    return ret;
}

std::string get_packets(std::string filename)
{
    std::ifstream infile("../test_suite/" + filename + ".txt");
    std::string line;
    std::string buffer;

    while (std::getline(infile, line))
    {
        if (line.length() > 0) {
            buffer.append(line);
        }
    }
    
    return buffer;
}

void print_nfa(std::vector<cuda_pair>* nfa)
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

void print_nfa(cuda_pair** nfa_cuda, int* size_cfg)
{
    for (int i = 0; i < 256; i++)
    {
        for (int j = 0; j < size_cfg[i]; j++)
        {
            cuda_pair transition = nfa_cuda[i][j];
            printf("%d - %d \n", transition.src, transition.dest);
        }
    }
}