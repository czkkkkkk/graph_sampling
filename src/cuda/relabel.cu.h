#pragma once
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

#include <thrust/binary_search.h>
#include <thrust/device_vector.h>

constexpr static const int BLOCK_SIZE = 256;
constexpr static const size_t TILE_SIZE = 1024;

template <typename IdType>
class HostOrderedHashTable;

template <typename IdType>
class DeviceOrderedHashTable
{
  public:
    friend class HostOrderedHashTable<IdType>;
    static constexpr IdType kEmptyKey = static_cast<IdType>(-1);
    /**
     * \brief An entry in the hashtable.
     */
    struct Mapping {
        /**
         * \brief The ID of the item inserted.
         */
        IdType key;
        /**
         * \brief The index of the item when inserted into the hashtable (e.g.,
         * the index within the array passed into FillWithDuplicates()).
         */
        int index;
        /**
         * \brief The index of the item in the unique list.
         */
        int local;
    };

    typedef const Mapping *ConstIterator;
    typedef Mapping *Iterator;

    DeviceOrderedHashTable(Mapping *table = nullptr, size_t size = 0)
        : table_(table), size_(size)
    {
    }

    /**
     * \brief Find the non-mutable mapping of a given key within the hash table.
     *
     * WARNING: The key must exist within the hashtable. Searching for a key not
     * in the hashtable is undefined behavior.
     *
     * \param id The key to search for.
     *
     * \return An iterator to the mapping.
     */
    inline __device__ Iterator Search(const IdType id)
    {
        const size_t pos = SearchForPosition(id);

        return &table_[pos];
    }

    /**
     * @brief Insert key-index pair into the hashtable.
     *
     * @param id The ID to insert.
     * @param index The index at which the ID occured.
     *
     * @return An iterator to inserted mapping.
     */
    inline __device__ Iterator Insert(const IdType id, const size_t index)
    {
        size_t pos = Hash(id);

        // linearly scan for an empty slot or matching entry
        IdType delta = 1;
        while (!AttemptInsertAt(pos, id, index)) {
            pos = Hash(pos + delta);
            delta += 1;
        }

        return GetMutable(pos);
    }

  protected:
    Mapping *table_;
    size_t size_;

    /**
     * \brief Search for an item in the hash table which is known to exist.
     *
     * WARNING: If the ID searched for does not exist within the hashtable, this
     * function will never return.
     *
     * \param id The ID of the item to search for.
     *
     * \return The the position of the item in the hashtable.
     */
    inline __device__ size_t SearchForPosition(const IdType id) const
    {
        IdType pos = Hash(id);

        // linearly scan for matching entry
        IdType delta = 1;
        while (table_[pos].key != id) {
            pos = Hash(pos + delta);
            delta += 1;
        }

        return pos;
    }

    inline __device__ bool AttemptInsertAt(const size_t pos, const IdType id,
                                           const size_t index)
    {
        using Type = unsigned long long int;
        const IdType key =
            atomicCAS(reinterpret_cast<Type *>(&GetMutable(pos)->key),
                      static_cast<Type>(kEmptyKey), static_cast<Type>(id));
        if (key == kEmptyKey || key == id) {
            // we either set a match key, or found a matching key, so then place
            // the minimum index in position. Match the type of atomicMin, so
            // ignore linting
            atomicMin(
                reinterpret_cast<Type *>(&GetMutable(pos)->index),  // NOLINT
                static_cast<Type>(index));                          // NOLINT
            return true;
        } else {
            // we need to search elsewhere
            return false;
        }
    }

    inline __device__ Mapping *GetMutable(const size_t pos)
    {
        assert(pos < this->size_);
        // The parent class Device is read-only, but we ensure this can only be
        // constructed from a mutable version of OrderedHashTable, making this
        // a safe cast to perform.
        return this->table_ + pos;
    }

    /**
     * \brief Hash an ID to a to a position in the hash table.
     *
     * \param id The ID to hash.
     *
     * \return The hash.
     */
    inline __device__ size_t Hash(const IdType id) const { return id % size_; }
};

template <typename IdType>
class HostOrderedHashTable
{
  public:
    using Mapping = typename DeviceOrderedHashTable<IdType>::Mapping;
    static constexpr IdType kEmptyKey = static_cast<IdType>(-1);
    // Must be uniform bytes for memset to work
    HostOrderedHashTable(size_t num, int scale)
    {
        const size_t next_pow2 =
            1 << static_cast<size_t>(1 + std::log2(num >> 1));
        auto size = next_pow2 << scale;
        void *p;
        cudaMalloc(&p, size * sizeof(Mapping));
        cudaMemset(p, kEmptyKey, size * sizeof(Mapping));
        device_table_ = DeviceOrderedHashTable<IdType>(
            reinterpret_cast<Mapping *>(p), size);
    }
    ~HostOrderedHashTable() { cudaFree(device_table_.table_); }
    DeviceOrderedHashTable<IdType> DeviceHandle() { return device_table_; }

  private:
    DeviceOrderedHashTable<IdType> device_table_;
};

template <typename IdType, int BLOCK_SIZE, size_t TILE_SIZE>
__global__ void
generate_hashmap_duplicates(const IdType *const items, const int64_t num_items,
                            DeviceOrderedHashTable<IdType> table)
{
    assert(BLOCK_SIZE == blockDim.x);

    const size_t block_start = TILE_SIZE * blockIdx.x;
    const size_t block_end = TILE_SIZE * (blockIdx.x + 1);

#pragma unroll
    for (size_t index = threadIdx.x + block_start; index < block_end;
         index += BLOCK_SIZE) {
        if (index < num_items) { table.Insert(items[index], index); }
    }
}

template <typename IdType>
HostOrderedHashTable<IdType> *
FillWithDuplicates(const IdType *const input, const size_t num_input,
                   thrust::device_vector<IdType> &unique_items)
{
    const int64_t num_tiles = (num_input + TILE_SIZE - 1) / TILE_SIZE;

    const dim3 grid(num_tiles);
    const dim3 block(BLOCK_SIZE);

    auto host_table = new HostOrderedHashTable<IdType>(num_input, 1);
    DeviceOrderedHashTable<IdType> device_table = host_table->DeviceHandle();

    generate_hashmap_duplicates<IdType, BLOCK_SIZE, TILE_SIZE>
        <<<grid, block>>>(input, num_input, device_table);
    thrust::device_vector<int> item_prefix(num_input + 1, 0);

    using it = thrust::counting_iterator<IdType>;
    using Mapping = typename DeviceOrderedHashTable<IdType>::Mapping;
    thrust::for_each(it(0), it(num_input),
                     [count = thrust::raw_pointer_cast(item_prefix.data()),
                      table = device_table,
                      in = input] __device__(IdType i) mutable {
                         Mapping &mapping = *(table.Search(in[i]));
                         if (mapping.index == i) { count[i] = 1; }
                     });
    thrust::exclusive_scan(item_prefix.begin(), item_prefix.end(),
                           item_prefix.begin());
    size_t tot = item_prefix[num_input];
    unique_items.resize(tot);

    thrust::for_each(it(0), it(num_input),
                     [prefix = thrust::raw_pointer_cast(item_prefix.data()),
                      table = device_table, in = input,
                      u = thrust::raw_pointer_cast(
                          unique_items.data())] __device__(IdType i) mutable {
                         Mapping &mapping = *(table.Search(in[i]));
                         if (mapping.index == i) {
                             mapping.local = prefix[i]; //prefix[i] 和 i有什么不一样
                             u[prefix[i]] = in[i];
                         }
                     });
    return host_table;
}