#include <string>
#include <iostream>

int* build_next(std::string pat)
{
    int* ret = (int*) calloc(pat.size(), sizeof(int));
    ret[0] = 0;
    int prefix_len = 0;
    int i = 1;
    while (i < pat.size())
    {
        if (pat[prefix_len] == pat[i])
        {
            prefix_len += 1;
            ret[i] = prefix_len;
            i += 1;
        }
        else
        {
            if (prefix_len == 0)
            {
                ret[i] = 0;
                i += 1;
            }
            else
            {
                prefix_len = ret[prefix_len - 1];
            }
        }
    }

    return ret;
}

int kmp_search(std::string str, std::string pat)
{
    int* next = build_next(pat);

    int i = 0;
    int j = 0;
    while (i < str.size())
    {
        if (str[i] == pat[j])
        {
            i += 1;
            j += 1;
        }
        else if (j > 0)
        {
            j = next[j-1];
        }
        else
        {
            i += 1;
        }

        if (j == pat.size())
        {
            return i-j;
        }
    }

    return -1;
}