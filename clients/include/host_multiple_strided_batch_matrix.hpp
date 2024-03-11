/* ************************************************************************
 * Copyright (C) 2018-2024 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell cop-
 * ies of the Software, and to permit persons to whom the Software is furnished
 * to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IM-
 * PLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNE-
 * CTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * ************************************************************************ */

#pragma once

#include "host_alloc.hpp"
#include <iomanip>

//
// Local declaration of the device strided batch matrix.
//
template <typename T>
class device_multiple_strided_batch_matrix;

//!
//! @brief Implementation of a host strided batched matrix.
//!
template <typename T>
class host_multiple_strided_batch_matrix
{
public:
    //!
    //! @brief Disallow copying.
    //!
    host_multiple_strided_batch_matrix(const host_multiple_strided_batch_matrix&) = delete;

    //!
    //! @brief Disallow assigning.
    //!
    host_multiple_strided_batch_matrix& operator=(const host_multiple_strided_batch_matrix&)
        = delete;

    //!
    //! @brief Constructor.
    //! @param m           The number of rows of the Matrix.
    //! @param n           The number of cols of the Matrix.
    //! @param lda         The leading dimension of the Matrix.
    //! @param stride The stride.
    //! @param batch_count The batch count.
    //! @param multiple_count The batch count.
    //!
    explicit host_multiple_strided_batch_matrix(size_t         m,
                                                size_t         n,
                                                size_t         lda,
                                                rocblas_stride stride,
                                                int64_t        batch_count,
                                                int64_t        multiple_count)
        : m_m(m)
        , m_n(n)
        , m_lda(lda)
        , m_stride(stride)
        , m_batch_count(batch_count)
        , m_multiple_count(multiple_count)
        , m_multiple_stride(calculate_multiple_stride(n, lda, stride, batch_count))
        , m_nmemb(calculate_nmemb(n, lda, stride, batch_count, multiple_count))
    {
        bool valid_parameters = this->m_nmemb > 0;
        if(valid_parameters)
        {
            this->m_data = (T*)host_calloc_throw(this->m_nmemb, sizeof(T));
        }
    }

    //!
    //! @brief Destructor.
    //!
    ~host_multiple_strided_batch_matrix()
    {
        if(nullptr != this->m_data)
        {
            free(this->m_data);
            this->m_data = nullptr;
        }
    }

    //!
    //! @brief Returns the data pointer.
    //!
    T* data()
    {
        return this->m_data;
    }

    //!
    //! @brief Returns the data pointer.
    //!
    const T* data() const
    {
        return this->m_data;
    }

    //!
    //! @brief Returns the rows of the Matrix.
    //!
    size_t m() const
    {
        return this->m_m;
    }

    //!
    //! @brief Returns the cols of the Matrix.
    //!
    size_t n() const
    {
        return this->m_n;
    }

    //!
    //! @brief Returns the leading dimension of the Matrix.
    //!
    size_t lda() const
    {
        return this->m_lda;
    }

    //!
    //! @brief Returns the batch count.
    //!
    int64_t batch_count() const
    {
        return this->m_batch_count;
    }

    //!
    //! @brief Returns the multiple count.
    //!
    int64_t multiple_count() const
    {
        return this->m_multiple_count;
    }

    //!
    //! @brief Returns the stride.
    //!
    rocblas_stride stride() const
    {
        return this->m_stride;
    }

    //!
    //! @brief Returns the multiple stride.
    //!
    rocblas_stride multiple_stride() const
    {
        return this->m_multiple_stride;
    }

    //!
    //! @brief Returns nmemb.
    //!
    size_t nmemb() const
    {
        return this->m_nmemb;
    }

    //!
    //! @brief Returns pointer.
    //! @param multiple_index The batch index.
    //! @return A mutable pointer to the multiple_index'th matrix.
    //!
    T* operator[](int64_t multiple_index)
    {

        return this->m_data + this->m_multiple_stride * multiple_index;
    }

    //!
    //! @brief Returns non-mutable pointer.
    //! @param multiple_index The batch index.
    //! @return A non-mutable mutable pointer to the multiple_index'th matrix.
    //!
    const T* operator[](int64_t multiple_index) const
    {
        return this->m_data + this->m_multiple_stride * multiple_index;
    }

    //!
    //! @brief Cast operator.
    //! @remark Returns the pointer of the first matrix.
    //!
    operator T*()
    {
        return (*this)[0];
    }

    //!
    //! @brief Non-mutable cast operator.
    //! @remark Returns the non-mutable pointer of the first matrix.
    //!
    operator const T*() const
    {
        return (*this)[0];
    }

    //!
    //! @brief Tell whether resource allocation failed.
    //!
    explicit operator bool() const
    {
        return nullptr != this->m_data;
    }

    //!
    //! @brief Copy data from a multiple strided batched matrix on host.
    //! @param that That strided batched matrix on host.
    //! @return true if successful, false otherwise.
    //!
    bool copy_from(const host_multiple_strided_batch_matrix& that)
    {
        if(that.m() == this->m_m && that.n() == this->m_n && that.lda() == this->m_lda
           && that.stride() == this->m_stride && that.batch_count() == this->m_batch_count
           && that.multiple_stride() == this->m_multiple_stride
           && that.multiple_count() == this->m_multiple_count)
        {
            memcpy(this->data(), that.data(), sizeof(T) * this->m_nmemb);
            return true;
        }
        else
        {
            return false;
        }
    }

    //!
    //! @brief Transfer data from a multiple strided batched matrix on device.
    //! @param that That strided batched matrix on device.
    //! @return The hip error.
    //!
    hipError_t transfer_from(const device_strided_batch_matrix<T>& that)
    {
        hipError_t hip_err;

        if(that.use_HMM && hipSuccess != (hip_err = hipDeviceSynchronize()))
            return hip_err;

        return hipMemcpy(this->m_data,
                         that.data(),
                         sizeof(T) * this->m_nmemb,
                         that.use_HMM ? hipMemcpyHostToHost : hipMemcpyDeviceToHost);
    }

    //!
    //! @brief Transfer data from a multiple strided batched matrix on device.
    //! @param that That multiple strided batched matrix on device.
    //! @return The hip error.
    //!
    hipError_t transfer_from(const device_multiple_strided_batch_matrix<T>& that)
    {
        hipError_t hip_err;

        if(that.use_HMM && hipSuccess != (hip_err = hipDeviceSynchronize()))
            return hip_err;

        return hipMemcpy(this->m_data,
                         that.data(),
                         sizeof(T) * this->m_nmemb,
                         that.use_HMM ? hipMemcpyHostToHost : hipMemcpyDeviceToHost);
    }

    //!
    //! @brief Check if memory exists.
    //! @return hipSuccess if memory exists, hipErrorOutOfMemory otherwise.
    //!
    hipError_t memcheck() const
    {
        return ((bool)*this) ? hipSuccess : hipErrorOutOfMemory;
    }

private:
    size_t         m_m{};
    size_t         m_n{};
    size_t         m_lda{};
    rocblas_stride m_stride{};
    int64_t        m_batch_count{};
    int64_t        m_multiple_count{};
    int64_t        m_multiple_stride{};
    size_t         m_nmemb{};
    T*             m_data{};

    static int64_t
        calculate_multiple_stride(size_t n, size_t lda, rocblas_stride stride, int64_t batch_count)
    {
        return align_stride<T>(lda * n + size_t(batch_count - 1) * std::abs(stride));
    }

    static size_t calculate_nmemb(
        size_t n, size_t lda, rocblas_stride stride, int64_t batch_count, int64_t multiple_count)
    {
        int64_t multiple_stride = calculate_multiple_stride(n, lda, stride, batch_count);
        return multiple_stride * multiple_count;
    }
};
