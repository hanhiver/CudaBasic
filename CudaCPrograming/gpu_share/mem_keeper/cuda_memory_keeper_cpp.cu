#include <iostream> 
#include <vector> 
#include <chrono> 
#include <thread> 

class memory_keeper
{
private: 
    std::vector<void*> _memory;
    const size_t _block_size = 128 * 1024 * 1024; //128MB. 
    size_t _blocks;  

    void allocate_block()
    {
        void *block; 
        cudaMalloc(&block, _block_size); 
        _memory.push_back(block); 
        ++ _blocks; 
    }

    void free_block()
    {
        void *block = _memory.back();
        cudaFree(block); 
        _memory.pop_back();
        -- _blocks;
    }

public: 
    memory_keeper() : _blocks {0}
    {}

    ~memory_keeper()
    {
        if (!_memory.empty())
        {
            void *block = _memory.front();
            cudaFree(block); 
        }
        _blocks = 0; 
    }

    size_t get_blocks()
    {
        return _blocks; 
    }

    bool allocate(size_t blocks)
    {
        for (size_t i=0; i<blocks; ++i)
        {
            allocate_block(); 
        }
        return true; 
    }

    bool free(size_t blocks)
    {
        if (blocks > _blocks)
        {
            return false; 
        }
        
        for (size_t i=0; i<blocks; ++i)
        {
            free_block();
        }

        return true; 
    }

    bool reallocate()
    {
        size_t original_blocks = _blocks;

        while (_blocks > 0)
        {
            free_block();
        }

        for (size_t i=0; i<original_blocks; ++i)
        {
            allocate_block(); 
        }
        return true; 
    }
};

int main()
{
    memory_keeper mk; 
    //c++17, no support. using namespace std::chrono_literals; 
    std::cout << "Try to allocate 1G mem: " << std::endl; 
    mk.allocate(8); 
    std::cout << "Done, sleep for 5 seconds. " << std::endl; 
    std::this_thread::sleep_for(std::chrono::seconds(5));
    std::cout << "Release 512M mem: " << std::endl; 
    mk.free(4); 
    std::cout << "Done, sleep for 5 seconds. " << std::endl; 
    std::this_thread::sleep_for(std::chrono::seconds(5));
    std::cout << "Try to allocate another 1G mem: " << std::endl;
    mk.allocate(8);
    std::cout << "Done, sleep for 5 seconds. " << std::endl; 
    std::this_thread::sleep_for(std::chrono::seconds(5));
    std::cout << "Clean them all. " << std::endl; 
}