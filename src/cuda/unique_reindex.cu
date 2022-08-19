#include "cuda_ops.cuh"
#include "atomic.cuh"


template <typename IdType>
struct relabel_hashmap
{
    __device__ inline relabel_hashmap(IdType* Kptr, IdType* Vptr, size_t numel) :
        kptr(Kptr),
        vptr(Vptr),
        capacity(numel) {};

    __device__ inline void Update(IdType key, IdType value){
        uint32_t delta = 1;
        uint32_t pos = hash(key);
        IdType prev = AtomicCAS(&kptr[pos], kEmptyKey, key);

        while(prev != key and prev != kEmptyKey){
            pos = hash(pos + delta);
            delta += 1;
            prev = AtomicCAS(&kptr[pos], kEmptyKey, key);
        }

        AtomicMin(vptr + pos, value);
    }


    __device__ inline IdType SearchForPos(IdType key){
        uint32_t delta = 1;
        uint32_t pos = hash(key);
    
        while(true){
            if(kptr[pos] == key){
                return pos;
            }
            if(kptr[pos] == kEmptyKey){
                return -1;
            }
            pos = hash(pos + delta);
            delta += 1;
        }
    }


    __device__ inline IdType SearchForValue(IdType key){
        uint32_t delta = 1;
        uint32_t pos = hash(key);
    
        while(true){
            if(kptr[pos] == key){
                return vptr[pos];
            };
            if(kptr[pos] == kEmptyKey){
                return -1;
            }
            pos = hash(pos + delta);
            delta += 1;
        }
    }

    __device__ inline uint32_t hash(int32_t key){
        return key & (capacity - 1);
    }

    __device__ inline uint32_t hash(uint32_t key){
        return key & (capacity - 1);
    }

    __device__ inline uint32_t hash(int64_t key){
        return key & (capacity - 1);
    }

    __device__ inline uint32_t hash(uint64_t key){
        return key & (capacity - 1);
    }

    IdType kEmptyKey{-1};
    IdType* kptr;
    IdType* vptr;
    uint32_t capacity{0};
};


inline int UpPower(int key){
    int ret =  1 << static_cast<uint32_t>(std::log2(key) + 1);
    return ret;
}


// relabel
template<typename IdType, bool need_cached>
inline std::vector<torch::Tensor> unique_core(
    torch::Tensor total_tensor
){

    int num_items = total_tensor.numel();
    int dir_size = UpPower(num_items);


    IdType MAX = std::numeric_limits<IdType>::max();
    torch::Tensor key_tensor = torch::full({dir_size,}, -1, total_tensor.options());
    torch::Tensor index_tensor = torch::full({dir_size,}, MAX, total_tensor.options());

    // insert
    using it = thrust::counting_iterator<IdType>;
    thrust::for_each(it(0), it(num_items),
            [key = key_tensor.data_ptr<IdType>(), 
             index = index_tensor.data_ptr<IdType>(),
             in = total_tensor.data_ptr<IdType>(), 
             num_items,
             dir_size
            ] __device__(IdType i) mutable {
             relabel_hashmap<IdType> table(key, index, dir_size);
             table.Update(in[i], i);
         }
    );


    // prefix sum
    c10::Allocator* cuda_allocator =  ogs::get_cuda_allocator();
    c10::DataPtr item_prefix_data = cuda_allocator->allocate((num_items + 1) * sizeof(IdType));
    thrust::device_ptr<IdType> item_prefix(static_cast<IdType*>(item_prefix_data.get()));
    thrust::fill(item_prefix, item_prefix + num_items + 1, (IdType) 0);
    thrust::for_each(it(0), it(num_items),
            [  key = key_tensor.data_ptr<IdType>(), 
               index = index_tensor.data_ptr<IdType>(),
               in = total_tensor.data_ptr<IdType>(),
               count = thrust::raw_pointer_cast(item_prefix),
               num_items,
               dir_size
            ] __device__(IdType i) mutable {
                relabel_hashmap<IdType> table(key, index, dir_size);
                if(table.SearchForValue(in[i]) == i){
                    count[i] = 1;
                }
            }
    );
    cub_exclusiveSum<IdType>(thrust::raw_pointer_cast(item_prefix), num_items + 1);

    // unique
    int tot = item_prefix[num_items];
    torch::Tensor unique_tensor = torch::zeros({tot,}, total_tensor.options());

    torch::Tensor value_tensor;
    if(need_cached){
       value_tensor = torch::full({dir_size,}, -1, total_tensor.options());
    }


    thrust::for_each(it(0), it(num_items),
            [ key = key_tensor.data_ptr<IdType>(), 
              index = index_tensor.data_ptr<IdType>(),
              in = total_tensor.data_ptr<IdType>(),
              prefix = thrust::raw_pointer_cast(item_prefix),
              unique = unique_tensor.data_ptr<IdType>(),
              cache_value = need_cached ? value_tensor.data_ptr<IdType>() : nullptr,
              num_items,
              dir_size
            ] __device__(IdType i) mutable {
                    relabel_hashmap<IdType> table(key, index, dir_size);
                    IdType pos = table.SearchForPos(in[i]);
                    if(index[pos] == i){
                        unique[prefix[i]] = in[i];
                        if(cache_value){
                            cache_value[pos] = prefix[i];
                        }
                    }
                }
            );

    if(need_cached){
        return {unique_tensor, key_tensor, value_tensor};
    } else {
        return {unique_tensor};
    }
}


template<typename IdType>
inline torch::Tensor relabel_core(
    torch::Tensor total_tensor,
    torch::Tensor key_tensor,
    torch::Tensor value_tensor
){
    int num_items = total_tensor.numel();
    using it = thrust::counting_iterator<IdType>;
    torch::Tensor relabel_tensor = torch::zeros_like(total_tensor);
    int dir_size = UpPower(num_items);

    thrust::for_each(it(0), it(num_items),
            [ key = key_tensor.data_ptr<IdType>(), 
              value = value_tensor.data_ptr<IdType>(),
              in = total_tensor.data_ptr<IdType>(),
              out = relabel_tensor.data_ptr<IdType>(),
              dir_size
            ] __device__(IdType i) mutable {
                    relabel_hashmap<IdType> table(key, value, dir_size);
                    out[i] = table.SearchForValue(in[i]);
                }
            );
    return relabel_tensor;
}



torch::Tensor unorder_unique(
    std::vector<torch::Tensor> data
){
    torch::Tensor total_tensor = torch::cat(data, 0);
    return std::get<0>(torch::_unique(total_tensor, false, false));
}

torch::Tensor unorder_unique_single(
    torch::Tensor data
){
    torch::Tensor total_tensor = data;
    return std::get<0>(torch::_unique(total_tensor, false, false));
}

torch::Tensor unique(
    std::vector<torch::Tensor> data
){
    torch::Tensor total_tensor = torch::cat(data, 0);
    torch::Tensor ret_tensor;
    OGS_ID_TYPE_SWITCH(total_tensor.dtype(), IdType, {
        ret_tensor = unique_core<IdType, false>(total_tensor)[0];
    });
    return ret_tensor;
}

torch::Tensor unique_single(
    torch::Tensor data
){
    torch::Tensor total_tensor = data;
    torch::Tensor ret_tensor;
    OGS_ID_TYPE_SWITCH(total_tensor.dtype(), IdType, {
        ret_tensor = unique_core<IdType, false>(total_tensor)[0];
    });
    return ret_tensor;
}

std::vector<torch::Tensor> unique_with_cache(
    std::vector<torch::Tensor> data
){
    torch::Tensor total_tensor = torch::cat(data, 0);
    std::vector<torch::Tensor> ret;
    OGS_ID_TYPE_SWITCH(total_tensor.dtype(), IdType, {
        ret = unique_core<IdType, true>(total_tensor);
    });
    return ret;
}

std::vector<torch::Tensor> unique_with_cache_single(
    torch::Tensor data
){
    torch::Tensor total_tensor = data;
    std::vector<torch::Tensor> ret;
    OGS_ID_TYPE_SWITCH(total_tensor.dtype(), IdType, {
        ret = unique_core<IdType, true>(total_tensor);
    });
    return ret;
}

std::tuple<torch::Tensor, std::vector<torch::Tensor>> relabel(
    std::vector<torch::Tensor> data
){
    std::vector<int64_t> split_sizes;
    for(auto d : data){
        split_sizes.push_back(d.numel());
    }

    torch::Tensor total_tensor = torch::cat(data, 0);
    torch::Tensor unique_tensor;
    torch::Tensor reindex_tensor;
    OGS_ID_TYPE_SWITCH(total_tensor.dtype(), IdType, {
        std::vector<torch::Tensor> unique_result = unique_core<IdType, true>(total_tensor);
        unique_tensor = unique_result[0];
        reindex_tensor = relabel_core<IdType>(total_tensor, unique_result[1], unique_result[2]);
    });

    std::vector<torch::Tensor> ret = reindex_tensor.split_with_sizes(split_sizes, 0);
    return std::make_tuple(unique_tensor, ret);
}

std::tuple<torch::Tensor, torch::Tensor> relabel_single(
    torch::Tensor data
){
    
    torch::Tensor total_tensor = data;
    torch::Tensor unique_tensor;
    torch::Tensor reindex_tensor;
    OGS_ID_TYPE_SWITCH(total_tensor.dtype(), IdType, {
        std::vector<torch::Tensor> unique_result = unique_core<IdType, true>(total_tensor);
        unique_tensor = unique_result[0];
        reindex_tensor = relabel_core<IdType>(total_tensor, unique_result[1], unique_result[2]);
    });

    return std::make_tuple(unique_tensor, reindex_tensor);
}


std::vector<torch::Tensor> relabel_with_cache(
    std::vector<torch::Tensor> data,
    torch::Tensor key_tensor,
    torch::Tensor value_tensor
){
    std::vector<int64_t> split_sizes;
    for(auto d : data){
        split_sizes.push_back(d.numel());
    }

    torch::Tensor total_tensor = torch::cat(data, 0);
    torch::Tensor reindex_tensor;
    OGS_ID_TYPE_SWITCH(total_tensor.dtype(), IdType, {
        reindex_tensor = relabel_core<IdType>(total_tensor, key_tensor, value_tensor);
    });

    return reindex_tensor.split_with_sizes(split_sizes, 0);
}

torch::Tensor relabel_with_cache_single(
    torch::Tensor data,
    torch::Tensor key_tensor,
    torch::Tensor value_tensor
){

    torch::Tensor total_tensor = data;
    torch::Tensor reindex_tensor;
    OGS_ID_TYPE_SWITCH(total_tensor.dtype(), IdType, {
        reindex_tensor = relabel_core<IdType>(total_tensor, key_tensor, value_tensor);
    });

    return reindex_tensor;
}

static auto registry =
    torch::RegisterOperators(
        "ogs::unique(Tensor[] data) -> Tensor", &unique)
        .op("ogs::unique_single(Tensor data) -> Tensor", &unique_single)
        .op("ogs::unique_with_cache(Tensor[] data) -> Tensor[]", &unique_with_cache)
        .op("ogs::unique_with_cache_single(Tensor data) -> Tensor[]", &unique_with_cache_single)
        .op("ogs::relabel(Tensor[] data) -> (Tensor, Tensor[])", &relabel)
        .op("ogs::relabel_single(Tensor data) -> (Tensor, Tensor)", &relabel_single)
        .op("ogs::relabel_with_cache(Tensor[] data, Tensor key_tensor, Tensor value_tensor) -> Tensor[]", &relabel_with_cache)
        .op("ogs::relabel_with_cache_single(Tensor data, Tensor key_tensor, Tensor value_tensor) -> Tensor", &relabel_with_cache_single)
        .op("ogs::unorder_unique(Tensor[] data) -> Tensor", &unorder_unique)
        .op("ogs::unorder_unique_single(Tensor data) -> Tensor", &unorder_unique_single);