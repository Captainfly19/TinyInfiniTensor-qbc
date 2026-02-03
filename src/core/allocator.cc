#include "core/allocator.h"
#include <algorithm>
#include <cstdio>
#include <utility>

namespace infini
{
    Allocator::Allocator(Runtime runtime) : runtime(runtime)
    {
        used = 0;
        peak = 0;
        ptr = nullptr;

        // 'alignment' defaults to sizeof(uint64_t), because it is the length of
        // the longest data type currently supported by the DataType field of
        // the tensor
        alignment = sizeof(uint64_t);
    }

    Allocator::~Allocator()
    {
        if (this->ptr != nullptr)
        {
            runtime->dealloc(this->ptr);
        }
    }

    size_t Allocator::alloc(size_t size)
    {
        printf("function begin\r\n");
        IT_ASSERT(this->ptr == nullptr);
        // pad the size to the multiple of alignment

        printf("pad begin\r\n");
        size = this->getAlignedSize(size);

        //it遍历所有的空内存块
        auto it = free_block.begin();
        //first 起始地址/终止地址 second 块的大小
        while(it != free_block.end())
        {
            if(it->second >= size)
            {
                size_t addr = it -> first;

                //若空闲块的大小比需要分配的空间大，拆分空闲块
                if(it->second > size)
                {
                    free_block[addr+size] = it->second - size;
                }
                //移除已经分配的空闲块
                free_block.erase(it);

                used += size;
                peak = std::max(peak, used);
                return addr;
            }
            ++it;
        }
        printf("assign addr\r\n");

        size_t addr = used;
        used += size;
        peak = std::max(peak, used);

        printf("return addr %ld\r\n",addr);
        return addr;
        // =================================== 作业 ===================================
        // TODO: 设计一个算法来分配内存，返回起始地址偏移量
        // =================================== 作业 ===================================

        return 0;
    }

    void Allocator::free(size_t addr, size_t size)
    {
        IT_ASSERT(this->ptr == nullptr);
        size = getAlignedSize(size);

        auto it = free_block.lower_bound(addr);

        //合并前面的空闲块
        if(it != free_block.begin())
        {
            auto prev = std::prev(it);
            if(prev->first + prev->second >= addr)
            {
                prev->second += (size - (prev->first + prev->second - addr));
                addr = prev->first;
                used -= (size - (prev->first + prev->second - addr));
                return ;
            }
        }

        if(it != free_block.end() && addr + size >= it->first)
        {
            size += (it->second - (addr + size - it->first));
            used -= (size - it->second);
            free_block.erase(it);
            free_block[addr] = size;
            return;
        }

        free_block[addr] = size;

        used -= size;
        // =================================== 作业 ===================================
        // TODO: 设计一个算法来回收内存
        // =================================== 作业 ===================================
    }

    void *Allocator::getPtr()
    {
        if (this->ptr == nullptr)
        {
            this->ptr = runtime->alloc(this->peak);
            printf("Allocator really alloc: %p %lu bytes\n", this->ptr, peak);
        }
        return this->ptr;
    }

    size_t Allocator::getAlignedSize(size_t size)
    {
        return ((size - 1) / this->alignment + 1) * this->alignment;
    }

    void Allocator::info()
    {
        std::cout << "Used memory: " << this->used
                  << ", peak memory: " << this->peak << std::endl;
    }
}
