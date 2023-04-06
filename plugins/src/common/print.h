//
// Created by nrsl on 23-3-20.
//

#ifndef TENSORRT_PRINT_H
#define TENSORRT_PRINT_H
#include <cuda.h>
#include <iostream>
#include <vector>
template <class T>
void print(const T& ptr, const std::vector<int>& shape, const std::string& infos)
{
    int size = 1;

    for (auto&& x : shape)
    {
        size *= x;
    }
    std::vector<std::remove_pointer_t<T>> data(size);
    cudaDeviceSynchronize();
    cudaMemcpy(data.data(), ptr, sizeof(std::remove_pointer_t<T>) * size, cudaMemcpyDeviceToHost);
    std::cout << infos << ": " << std::endl;

    int len = 32000;
    int wrap = 100000;
    if (shape.size() > 1 and shape[shape.size() - 1] > 1)
    {
        size /= shape[shape.size() - 1];
        wrap = shape[shape.size() - 1];
    }

    std::cout << "[";
    if (size < len * 2)
    {
        for (int i = 0; i < data.size(); ++i)
        {
            std::cout << data[i] << (i < data.size() - 1 ? "," : "");
            if ((i + 1) % wrap == 0)
            {
                std::cout << "\n";
            }
        }
    }
    else
    {
        for (int i = 0; i < len; ++i)
        {
            std::cout << data[i] << ",";
        }
        std::cout << "...,";
        for (int i = data.size() - len - 1; i < data.size(); ++i)
        {
            std::cout << data[i] << (i < data.size() - 1 ? "," : "");
        }
    }
    std::cout << "]";
    std::cout << std::endl;
}
#endif // TENSORRT_PRINT_H
