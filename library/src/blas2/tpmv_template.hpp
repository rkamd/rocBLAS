/* ************************************************************************
 * Copyright 2019-2020 Advanced Micro Devices, Inc.
 * ************************************************************************ */
#pragma once
#include "../blas1/rocblas_copy.hpp"
#include "tpmv_device.hpp"

template <rocblas_int NB, typename A, typename X, typename W>
rocblas_status tpmv_template(rocblas_handle    handle,
                             rocblas_fill      uplo,
                             rocblas_operation transa,
                             rocblas_diagonal  diag,
                             rocblas_int       m,
                             A                 a,
                             ptrdiff_t         offseta,
                             rocblas_stride    stridea,
                             X                 x,
                             ptrdiff_t         offsetx,
                             rocblas_int       incx,
                             rocblas_stride    stridex,
                             W                 w,
                             rocblas_stride    stridew,
                             rocblas_int       batch_count)
{
    //
    // quick return
    //
    if(!m || !batch_count)
    {
        return rocblas_status_success;
    }

    hipStream_t rocblas_stream = handle->get_stream();

    ptrdiff_t shiftx = incx < 0 ? offsetx + ptrdiff_t(incx) * (1 - m) : offsetx;

    dim3 tpmv_grid((m - 1) / NB + 1, batch_count);
    dim3 tpmv_threads(NB);

    // Temporarily change the thread's default device ID to the handle's device ID
    auto saved_device_id = handle->push_device_id();

    switch(transa)
    {
    case rocblas_operation_none:
    {
        hipLaunchKernelGGL((tpmvn_kernel<NB>),
                           tpmv_grid,
                           tpmv_threads,
                           0,
                           rocblas_stream,
                           uplo,
                           diag,
                           m,
                           a,
                           offseta,
                           stridea,
                           x,
                           shiftx,
                           incx,
                           stridex,
                           w,
                           stridew);
        break;
    }

    case rocblas_operation_transpose:
    {
        hipLaunchKernelGGL((tpmvt_kernel<NB>),
                           tpmv_grid,
                           tpmv_threads,
                           0,
                           rocblas_stream,
                           uplo,
                           diag,
                           m,
                           a,
                           offseta,
                           stridea,
                           x,
                           shiftx,
                           incx,
                           stridex,
                           w,
                           stridew);
        break;
    }

    case rocblas_operation_conjugate_transpose:
    {
        hipLaunchKernelGGL((tpmvc_kernel<NB>),
                           tpmv_grid,
                           tpmv_threads,
                           0,
                           rocblas_stream,
                           uplo,
                           diag,
                           m,
                           a,
                           offseta,
                           stridea,
                           x,
                           shiftx,
                           incx,
                           stridex,
                           w,
                           stridew);

        break;
    }
    }

    //
    // Copy workspace to x.
    //
    {
        static constexpr rocblas_int offsetw = 0;
        static constexpr rocblas_int incw    = 1;
        return rocblas_copy_template<false, NB>(
            handle, m, w, offsetw, incw, stridew, x, offsetx, incx, stridex, batch_count);
    }
}
