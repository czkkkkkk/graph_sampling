#include "cuda_ops.cuh"
#include "atomic.cuh"
#include "utils.cuh"



int64_t UpPower(int64_t key){
    int64_t ret =  1 << static_cast<uint64_t>(std::log2(key) + 1);
    return ret;
}

/******************************* shifthash **************************************/
/**************** Thomas Wang's 32 bit and 64 bit Mix Function ******************/
__device__ inline uint32_t Hash32Shift(uint32_t key){
    key = ~key + (key << 15);               // # key = (key << 15) - key - 1;
    key = key ^ (key >> 12);
    key = key + (key << 2);
    key = key ^ (key >> 4);
    key = key * 2057;                       // key = (key + (key << 3)) + (key << 11);
    key = key ^ (key >> 16);
    return key;
}


__device__ inline uint64_t hash64shift(uint64_t key)
{
  key = (~key) + (key << 21);               // key = (key << 21) - key - 1;
  key = key ^ (key >> 24);
  key = (key + (key << 3)) + (key << 8);    // key * 265
  key = key ^ (key >> 14);
  key = (key + (key << 2)) + (key << 4);    // key * 21
  key = key ^ (key >> 28);
  key = key + (key << 31);
  return key;
}


template<typename K, typename V, typename Op>
struct StaticHashmap
{
    // Kptr, Vptr should be initialized before construct the StaticHashmap
    __device__ inline StaticHashmap(K* Kptr, K emptykey, V* Vptr, V defaultvalue, size_t dir_size) : 
        kEmptyKey(emptykey),
        DefaultValue(defaultvalue),
        kptr(Kptr),
        vptr(Vptr),
        capacity(dir_size) {}

    __device__ inline void Update(K key, V value){
        uint32_t delta = 1;
        uint32_t pos = hash(key);
        K prev = AtomicCAS(&kptr[pos], kEmptyKey, key);

        while(prev != key and prev != kEmptyKey){
            pos = hash(pos + delta);
            delta += 1;
            prev = AtomicCAS(&kptr[pos], kEmptyKey, key);
        }
        
        Op::Call(vptr + pos, value);
    }

    __device__ inline K SearchForPos(K key){
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
    

    __device__ inline V SearchForValue(K key){
        uint32_t delta = 1;
        uint32_t pos = hash(key);
    
        while(true){
            if(kptr[pos] == key){
                return vptr[pos];
            };
            if(kptr[pos] == kEmptyKey){
                return DefaultValue;
            }
            pos = hash(pos + delta);
            delta += 1;
        }
    }


    __device__ inline uint32_t hash(int32_t key){
        return Hash32Shift(key) & (capacity - 1);
    }

    __device__ inline uint32_t hash(uint32_t key){
        return Hash32Shift(key) & (capacity - 1);
    }

    __device__ inline uint32_t hash(int64_t key){
        return static_cast<uint32_t>(hash64shift(key)) & (capacity - 1);
    }

    __device__ inline uint32_t hash(uint64_t key){
        return static_cast<uint32_t>(hash64shift(key)) & (capacity - 1);
    }

    

    K kEmptyKey;
    V DefaultValue;
    K* kptr;
    V* vptr;
    uint32_t capacity{0};
};

template<typename KType, typename VType, typename Op>
void static_hashmap_update_core(
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor key_buffer,
    torch::Tensor value_buffer,
    int _default_value
){
    int num_items = key.numel();
    int dir_size = key_buffer.numel();
    CHECK(dir_size > num_items);
    
    KType empty_key = EmptyKey<KType>::getValue();
    VType default_value = get_default_value<VType>(_default_value);

    using it = thrust::counting_iterator<KType>;
    thrust::for_each(it(0), it(num_items),
        [ 
            _key = key.data_ptr<KType>(),
            _value = value.data_ptr<VType>(),
            _key_buffer = key_buffer.data_ptr<KType>(),
            _value_buffer = value_buffer.data_ptr<VType>(),
            empty_key, default_value, dir_size
        ]  __device__(KType i) mutable {
            StaticHashmap<KType, VType, Op> hashmap(
                _key_buffer, empty_key, _value_buffer, default_value, dir_size);

            hashmap.Update(_key[i], _value[i]);
    });
}



template<typename KType, typename VType, typename Op>
torch::Tensor static_hashmap_search_core(
    torch::Tensor key,
    torch::Tensor key_buffer,
    torch::Tensor value_buffer,
    int _default_value
){
    int num_items = key.numel();
    int dir_size = key_buffer.numel();
    
    KType empty_key = EmptyKey<KType>::getValue();
    VType default_value = get_default_value<VType>(_default_value);
    torch::Tensor value = torch::full({num_items,}, default_value, value_buffer.options());

    using it = thrust::counting_iterator<KType>;
    thrust::for_each(it(0), it(num_items),
        [ 
            _key = key.data_ptr<KType>(),
            _value = value.data_ptr<VType>(),
            _key_buffer = key_buffer.data_ptr<KType>(),
            _value_buffer = value_buffer.data_ptr<VType>(),
            empty_key, default_value, dir_size
        ]  __device__(KType i) mutable {
            StaticHashmap<KType, VType, Op> hashmap(
                _key_buffer, empty_key, _value_buffer, default_value, dir_size);

            _value[i] = hashmap.SearchForValue(_key[i]);
    });

    return value;
}

std::tuple<torch::Tensor, torch::Tensor> static_hashmap_update(
    torch::Tensor key,
    torch::Tensor value,
    torch::Tensor key_buffer,
    torch::Tensor value_buffer,
    int64_t _default_value,
    int64_t _op
){
    OGS_ID_TYPE_SWITCH(key_buffer.dtype(), KType, {
        OGS_VALUE_TYPE_SWITCH(value_buffer.dtype(), VType, {
            OGS_GPU_HASH_TABLE_OP(_op, VType, OpType, {
                static_hashmap_update_core<KType, VType, OpType>(
                    key, value, key_buffer, value_buffer, _default_value
                );
            });
        });
    });
    return std::make_tuple(key_buffer, value_buffer);
}

torch::Tensor static_hashmap_search(
    torch::Tensor key,
    torch::Tensor key_buffer,
    torch::Tensor value_buffer,
    int64_t _default_value
){
    torch::Tensor value;
    OGS_ID_TYPE_SWITCH(key_buffer.dtype(), KType, {
        OGS_VALUE_TYPE_SWITCH(value_buffer.dtype(), VType, {
            value = static_hashmap_search_core<KType, VType, Assign<VType>>(
                key, key_buffer, value_buffer, _default_value
            );
        });
    });
    return value; 
}



static auto registry =
    torch::RegisterOperators(
        "ogs::static_hashmap_update(Tensor key, Tensor value, Tensor key_buffer, Tensor value_buffer,"
            "int default_value, int op) -> (Tensor, Tensor)", &static_hashmap_update)
        .op("ogs::static_hashmap_search(Tensor key, Tensor key_buffer, Tensor value_buffer,"
            "int default_value) -> Tensor", &static_hashmap_search)
        .op("ogs::UpPower(int size) -> int", &UpPower);