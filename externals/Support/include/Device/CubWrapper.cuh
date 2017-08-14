/**
 * @internal
 * @brief Vec-Tree interface
 * @author Federico Busato                                                  <br>
 *         Univerity of Verona, Dept. of Computer Science                   <br>
 *         federico.busato@univr.it
 * @date August, 2017
 * @version v2
 *
 * @copyright Copyright © 2017 cuStinger. All rights reserved.
 *
 * @license{<blockquote>
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * * Redistributions of source code must retain the above copyright notice, this
 *   list of conditions and the following disclaimer.
 * * Redistributions in binary form must reproduce the above copyright notice,
 *   this list of conditions and the following disclaimer in the documentation
 *   and/or other materials provided with the distribution.
 * * Neither the name of the copyright holder nor the names of its
 *   contributors may be used to endorse or promote products derived from
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 * </blockquote>}
 *
 * @file
 */
#pragma once

#include <limits>

namespace xlib {

class CubWrapper {
protected:
    explicit CubWrapper(size_t num_items) noexcept;
    ~CubWrapper() noexcept;

    void*        _d_temp_storage     { nullptr };
    size_t       _temp_storage_bytes { 0 };
    const size_t _num_items;
};

template<typename T>
class CubSortByValue : public CubWrapper {
public:
    explicit CubSortByValue(const T* d_in, size_t size, T* d_sorted,
                            T d_in_max = std::numeric_limits<T>::max())
                            noexcept;
    void run() noexcept;

private:
    const T* _d_in;
    T*       _d_sorted;
    T        _d_in_max;
};

template<typename T, typename R>
class CubSortByKey : public CubWrapper {
public:
    CubSortByKey(const T* d_key, const R* d_data_in, size_t num_items,
                 T* d_key_sorted, R* d_data_out,
                 T d_key_max = std::numeric_limits<T>::max()) noexcept;

    void run() noexcept;

private:
    const T* _d_key;
    const R* _d_data_in;
    T*       _d_key_sorted;
    R*       _d_data_out;
    T        _d_key_max;
};

template<typename T, typename R>
class CubSortPairs2 : public CubWrapper {
public:
    CubSortPairs2(T* d_in1, R* d_in2, size_t num_items,
                  T d_in1_max = std::numeric_limits<T>::max(),
                  R d_in2_max = std::numeric_limits<R>::max());

    CubSortPairs2(T* d_in1, R* d_in2, size_t num_items,
                  T* d_in1_tmp, R* d_in2_tmp,
                  T d_in1_max = std::numeric_limits<T>::max(),
                  R d_in2_max = std::numeric_limits<R>::max());

    ~CubSortPairs2() noexcept;

    void run() noexcept;

private:
    T*       _d_in1, *_d_in1_tmp { nullptr };
    R*       _d_in2, *_d_in2_tmp { nullptr };
    T        _d_in1_max;
    R        _d_in2_max;
    bool     _internal_alloc { false };
};

template<typename T>
class CubUnique : public CubWrapper {
public:
    CubUnique(const T* d_in, size_t size, T*& d_unique_batch);
    ~CubUnique() noexcept;
    int run() noexcept;
private:
    const T* _d_in;
    T*&      _d_unique_batch;
    int*     _d_unique_egdes;
};

template<typename T, typename R = T>
class CubRunLengthEncode : public CubWrapper {
public:
    explicit CubRunLengthEncode(const T* d_in, size_t size,
                                T* d_unique_out, R* d_counts_out) noexcept;
    ~CubRunLengthEncode() noexcept;
    int run() noexcept;
private:
    const T* _d_in;
    T*       _d_unique_out;
    R*       _d_counts_out;
    R*       _d_num_runs_out  { nullptr };
};

template<typename T>
class PartitionFlagged : public CubWrapper {
public:
    explicit PartitionFlagged(const T* d_in, const bool* d_flags,
                              size_t num_items, T* d_out) noexcept;
    ~PartitionFlagged() noexcept;
    int run() noexcept;
private:
    const T*    _d_in;
    const bool* _d_flags;
    T*          _d_out;
    T*          _d_num_selected_out { nullptr };
};

template<typename T>
class CubReduce : public CubWrapper {
public:
    explicit CubReduce(const T* d_in, size_t size) noexcept;
    ~CubReduce() noexcept;
    T run() noexcept;
private:
    const T* _d_in  { nullptr };
    T*       _d_out { nullptr };
};

template<typename T>
class CubExclusiveSum : public CubWrapper {
public:
    explicit CubExclusiveSum(T* d_in_out, size_t size)             noexcept;
    explicit CubExclusiveSum(const T* d_in, size_t size, T* d_out) noexcept;
    void run() noexcept;
private:
    const T* _d_in;
    T*       _d_out;
};
/*
template<typename T>
class CubInclusiveSum : public CubWrapper {
public:
    CubInclusiveSum(T* d_in, size_t size);
    CubInclusiveSum(const T* d_in, size_t size, T*& d_out);
    ~CubInclusiveSum() noexcept;
    void run() noexcept;
private:
    static T* null_ptr_ref;
    const T*  _d_in;
    T*&       _d_out;
    T*        _d_in_out;
};
template<typename T>
T* CubInclusiveSum<T>::null_ptr_ref = nullptr;*/

template<typename T>
class CubPartitionFlagged : public CubWrapper {
public:
    CubPartitionFlagged(const T* d_in, const bool* d_flag, size_t size,
                        T*& d_out) noexcept;
    ~CubPartitionFlagged() noexcept;
    int run() noexcept;
    void run_no_copy() noexcept;
private:
    const T*    _d_in;
    T*&         _d_out;
    const bool* _d_flag;
    int*        _d_num_selected_out;
};

template<typename T>
class CubSelect : public CubWrapper {
public:
    CubSelect(T* d_in_out, size_t num_items) noexcept;
    CubSelect(const T* d_in, size_t num_items, T* d_out) noexcept;
    ~CubSelect() noexcept;

    int run_diff(const T& diff) noexcept;
private:
    const T* _d_in;
    T*       _d_out;
    int*     _d_num_selected_out;
    int      _num_items;
};

template<typename T>
class CubSelectFlagged : public CubWrapper {
public:
    CubSelectFlagged(T* d_in_out, size_t num_items, const bool* d_flags)
                     noexcept;
    CubSelectFlagged(const T* d_in, size_t num_items, const bool* d_flags,
                     T* d_out) noexcept;
    ~CubSelectFlagged() noexcept;

    int run() noexcept;
private:
    const T*    _d_in;
    T*          _d_out;
    int*        _d_num_selected_out;
    int         _num_items;
    const bool* _d_flags;
};

template<typename T>
class CubSegmentedReduce : public CubWrapper {
public:
    CubSegmentedReduce(int* _d_offsets, const T* d_in, int _num_segments,
                       T*& d_out);
    ~CubSegmentedReduce() noexcept;
    void run() noexcept;
private:
    int*  _d_offsets;
    const T*    _d_in;
    T*&         _d_out;
};

template<typename T>
class CubSpMV : public CubWrapper {
public:
    CubSpMV(T* d_value, int* d_row_offsets, int* d_column_indices,
            T* d_vector_x, T* d_vector_y,
            int num_rows, int num_cols, int num_nonzeros);
    //~CubSpMV() noexcept;
    void run() noexcept;
private:
    int*  _d_row_offsets;
    int*  _d_column_indices;
    T*    _d_vector_x;
    T*    _d_vector_y;
    T*    _d_values;
    int   _num_rows, _num_cols, _num_nonzeros;
};


template<typename T>
class CubArgMax : public CubWrapper {
public:
    explicit CubArgMax(const T* d_in, size_t size) noexcept;
    typename std::pair<int, T> run() noexcept;
private:
    const T* _d_in;
    void*    _d_out;
};

} // namespace xlib