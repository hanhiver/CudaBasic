#include <iostream> 
#include <queue> 
#include <chrono> 
#include <thread> 

class memory_keeper
{
private: 
    std::queue<void*> _memory;
    const size_t _block_size = 128 * 1024 * 1024; 
    size_t _blocks;  

    void allocate_block()
    {
        void *block; 
        cudaMalloc(&block, _block_size); 
        _memory.push(block); 
        ++ _blocks; 
    }

    void free_block()
    {
        void *block = _memory.front();
        cudaFree(block); 
        _memory.pop();
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
};

int main()
{
    memory_keeper mk; 
    //using namespace std::chrono_literals; 
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