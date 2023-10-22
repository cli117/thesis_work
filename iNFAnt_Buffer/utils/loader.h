#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <iostream>
#include <unordered_set>

struct cuda_pair
{
    int src;
    int dest;
};

inline bool ends_with(std::string const & value, std::string const & ending)
{
    if (ending.size() > value.size()) return false;
    return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

std::vector<cuda_pair>* get_nfa(std::string filename, std::unordered_set<int>* acc_set)
{
    std::vector<cuda_pair>* ret = (std::vector<cuda_pair>*) calloc(256, sizeof(std::vector<cuda_pair>));
    std::ifstream infile(filename);
    std::string line;
    bool first_skipped = false;
    while (std::getline(infile, line))
    {
        if (!first_skipped)
        {
            first_skipped = true;
            continue;
        }
        size_t dash_idx = line.find_first_of("-");
        size_t cln_idx = line.find_first_of(":");
        bool is_acc = ends_with(line, " acc");
        size_t end_idx = is_acc ? line.length() - 5 : line.length() - 1;
        int src = std::stoi(line.substr(0, dash_idx));
        int dest = std::stoi(line.substr(dash_idx+1, cln_idx - dash_idx - 1));
        if (is_acc)
        {
            acc_set->insert(dest);
        }

        if (src == dest){
            continue;
        }
        cuda_pair transition;
        transition.src = src;
        transition.dest = dest;
        char idx = line[cln_idx + 1];
        if (idx != '0')
        {
            continue;
        }
        std::string context = line.substr(cln_idx+2, end_idx - cln_idx);
        std::string hex_str = context.substr(1, 2);
        int hex_num = std::stoi(hex_str, nullptr, 16);
        ret[hex_num].push_back(transition);
    }
    return ret;
}

int get_num_of_states(std::string filename) 
{
    int ret = 0;
    std::ifstream infile(filename);
    std::string line;
    bool first_skipped = false;
    while (std::getline(infile, line))
    {
        if (!first_skipped)
        {
            first_skipped = true;
            continue;
        }
        size_t dash_idx = line.find_first_of("-");
        size_t cln_idx = line.find_first_of(":");
        int dest = std::stoi(line.substr(dash_idx+1, cln_idx - dash_idx - 1));
        ret = std::max(dest, ret);
    }

    return ret;
}

int* get_persistent_sv(std::string filename)
{
    int size = get_num_of_states(filename);
    int* ret = (int*) calloc(size, sizeof(int));
    std::ifstream infile(filename);
    std::string line;
    bool first_skipped = false;
    while (std::getline(infile, line))
    {
        if (!first_skipped)
        {
            first_skipped = true;
            continue;
        }
        size_t dash_idx = line.find_first_of("-");
        size_t cln_idx = line.find_first_of(":");

        int src = std::stoi(line.substr(0, dash_idx));
        int dest = std::stoi(line.substr(dash_idx+1, cln_idx - dash_idx - 1));
        if (src == dest){
            ret[src - 1] = true;
        }
    }

    return ret;
}

std::string get_longest_literal(std::string filename, int& size)
{
    std::ifstream infile(filename);
    std::string line;
    if (std::getline(infile, line))
    {
        size = line.size();
        return line;
    }

    return "";
}

std::vector<std::string> get_packets(std::string filename)
{
    std::vector<std::string> ret;
    std::ifstream infile(filename);
    std::string line;
    std::string buffer;
    int MIN_LENGTH = 10000;
    int MAX_LENGTH = 20000;
    while (std::getline(infile, line))
    {
        if (line.length() > 0) {
            if (line.length() > MAX_LENGTH) {
                // buffer have have content in it already
                while (line.length() > MAX_LENGTH) {
                    int add_length = MAX_LENGTH - buffer.length();
                    buffer.append(line.substr(0, add_length));
                    line = line.substr(add_length);
                    ret.push_back(buffer);
                    buffer = "";
                }
                ret.push_back(line);
            } else {
                buffer.append(line);
                if (buffer.length() < MIN_LENGTH) {
                    continue;
                }
                ret.push_back(buffer);
                buffer = "";
            }
        }
    }

    if (buffer.length() > 0) {
        ret.push_back(buffer);
    }

    // while (std::getline(infile, line))
    // {
    //     if (line.length() > 0) {
    //         ret.push_back(line);
    //     }
    // }
    
    return ret;
}

void print_nfa(std::vector<cuda_pair>* nfa)
{
    for (int i = 0; i < 256; i++)
    {
        for (int j = 0; j < nfa[i].size(); j++)
        {
            cuda_pair transition = nfa[i][j];
            printf("%dth char: %d - %d\n", i, transition.src, transition.dest);
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