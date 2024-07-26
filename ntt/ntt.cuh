// Copyright Supranational LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef __SPPARK_NTT_NTT_CUH__
#define __SPPARK_NTT_NTT_CUH__

#include <cassert>
#include <iostream>

#include <util/exception.cuh>
#include <util/rusterror.h>
#include <util/gpu_t.cuh>

#include "parameters.cuh"
#include "kernels.cu"

#ifndef __CUDA_ARCH__
float measure_time(cudaEvent_t start, cudaEvent_t stop) {
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  return milliseconds;
}
class NTT {
public:
    enum class InputOutputOrder { NN, NR, RN, RR };
    enum class Direction { forward, inverse };
    enum class Type { standard, coset };
    enum class Algorithm { GS, CT };

protected:
    static void bit_rev(fr_t* d_out, const fr_t* d_inp,
                        uint32_t lg_domain_size, stream_t& stream)
    {
        assert(lg_domain_size <= MAX_LG_DOMAIN_SIZE);

        size_t domain_size = (size_t)1 << lg_domain_size;
        // aim to read 4 cache lines of consecutive data per read
        const uint32_t Z_COUNT = 256 / sizeof(fr_t);
        const uint32_t bsize = Z_COUNT>WARP_SZ ? Z_COUNT : WARP_SZ;

        if (domain_size <= 1024)
            bit_rev_permutation<<<1, domain_size, 0, stream>>>
                               (d_out, d_inp, lg_domain_size);
        else if (domain_size < bsize * Z_COUNT)
            bit_rev_permutation<<<domain_size / WARP_SZ, WARP_SZ, 0, stream>>>
                               (d_out, d_inp, lg_domain_size);
        else if (Z_COUNT > WARP_SZ || lg_domain_size <= 32)
            bit_rev_permutation_z<<<domain_size / Z_COUNT / bsize, bsize,
                                    bsize * Z_COUNT * sizeof(fr_t), stream>>>
                                 (d_out, d_inp, lg_domain_size);
        else
            // Those GPUs that can reserve 96KB of shared memory can
            // schedule 2 blocks to each SM...
            bit_rev_permutation_z<<<gpu_props(stream).multiProcessorCount*2, 192,
                                    192 * Z_COUNT * sizeof(fr_t), stream>>>
                                 (d_out, d_inp, lg_domain_size);

        CUDA_OK(cudaGetLastError());
    }

private:
    static void LDE_powers(fr_t* inout, bool innt, bool bitrev,
                           uint32_t lg_domain_size, uint32_t lg_blowup,
                           stream_t& stream, bool ext_pow = false)
    {
        size_t domain_size = (size_t)1 << lg_domain_size;
        const auto gen_powers =
            NTTParameters::all(innt)[stream]->partial_group_gen_powers;

        if (domain_size < WARP_SZ)
            LDE_distribute_powers<<<1, domain_size, 0, stream>>>
                                 (inout, lg_blowup, bitrev, gen_powers, ext_pow);
        else if (domain_size < 512)
            LDE_distribute_powers<<<domain_size / WARP_SZ, WARP_SZ, 0, stream>>>
                                 (inout, lg_blowup, bitrev, gen_powers, ext_pow);
        else
            LDE_distribute_powers<<<domain_size / 512, 512, 0, stream>>>
                                 (inout, lg_blowup, bitrev, gen_powers, ext_pow);

        CUDA_OK(cudaGetLastError());
    }

protected:
    // coset_ext_pow is only used when NTT type is coset
    static void NTT_internal(fr_t* d_inout, uint32_t lg_domain_size,
                             InputOutputOrder order, Direction direction,
                             Type type, stream_t& stream,
                             bool coset_ext_pow = false)
    {
        // Pick an NTT algorithm based on the input order and the desired output
        // order of the data. In certain cases, bit reversal can be avoided which
        // results in a considerable performance gain.

        const bool intt = direction == Direction::inverse;
        const auto& ntt_parameters = *NTTParameters::all(intt)[stream];
        bool bitrev;
        Algorithm algorithm;

        switch (order) {
            case InputOutputOrder::NN:
                bit_rev(d_inout, d_inout, lg_domain_size, stream);
                bitrev = true;
                algorithm = Algorithm::CT;
                break;
            case InputOutputOrder::NR:
                bitrev = false;
                algorithm = Algorithm::GS;
                break;
            case InputOutputOrder::RN:
                bitrev = true;
                algorithm = Algorithm::CT;
                break;
            case InputOutputOrder::RR:
                bitrev = true;
                algorithm = Algorithm::GS;
                break;
            default:
                assert(false);
        }

        if (!intt && type == Type::coset)
            LDE_powers(d_inout, intt, bitrev, lg_domain_size, 0, stream,
                       coset_ext_pow);

        switch (algorithm) {
            case Algorithm::GS:
                GS_NTT(d_inout, lg_domain_size, intt, ntt_parameters, stream);
                break;
            case Algorithm::CT:
                CT_NTT(d_inout, lg_domain_size, intt, ntt_parameters, stream);
                break;
        }

        if (intt && type == Type::coset)
            LDE_powers(d_inout, intt, !bitrev, lg_domain_size, 0, stream,
                       coset_ext_pow);

        if (order == InputOutputOrder::RR)
            bit_rev(d_inout, d_inout, lg_domain_size, stream);
    }

public:
    static RustError Base(const gpu_t& gpu, fr_t* inout, uint32_t lg_domain_size,
                          InputOutputOrder order, Direction direction,
                          Type type, bool coset_ext_pow = false)
    {
        if (lg_domain_size == 0)
            return RustError{cudaSuccess};

        try {
            gpu.select();

            size_t domain_size = (size_t)1 << lg_domain_size;
            dev_ptr_t<fr_t> d_inout{domain_size, gpu};
            gpu.HtoD(&d_inout[0], inout, domain_size);

          const int iterations = 10;
          float total_time_ntt = 0.0f;
          cudaEvent_t start, stop;
          cudaEventCreate(&start);
          cudaEventCreate(&stop);
          for (int i = 0; i < iterations; ++i) {
            cudaEventRecord(start, 0);
            NTT_internal(&d_inout[0], lg_domain_size, order, direction, type, gpu,
                         coset_ext_pow);
            gpu.sync();
            cudaEventRecord(stop, 0);
            cudaEventSynchronize(stop);

            total_time_ntt += measure_time(start, stop);
          }

          float avg_time_ntt = total_time_ntt / iterations;
          std::cout << "Average time for NTT::Base: " << avg_time_ntt << " ms" << std::endl;

          cudaEventDestroy(start);
          cudaEventDestroy(stop);

            gpu.DtoH(inout, &d_inout[0], domain_size);
            gpu.sync();
        } catch (const cuda_error& e) {
            gpu.sync();
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
            return RustError{e.code(), e.what()};
#else
            return RustError{e.code()};
#endif
        }

        return RustError{cudaSuccess};
    }

    static RustError LDE(const gpu_t& gpu, fr_t* inout,
                         uint32_t lg_domain_size, uint32_t lg_blowup,
                         bool ext_pow = false)
    {
        try {
            gpu.select();

            size_t domain_size = (size_t)1 << lg_domain_size;
            size_t ext_domain_size = domain_size << lg_blowup;
            dev_ptr_t<fr_t> d_ext_domain{ext_domain_size, gpu};
            fr_t* d_domain = &d_ext_domain[ext_domain_size - domain_size];

            gpu.HtoD(&d_domain[0], inout, domain_size);

            NTT_internal(&d_domain[0], lg_domain_size,
                         InputOutputOrder::NR, Direction::inverse,
                         Type::standard, gpu);

            const auto gen_powers =
                NTTParameters::all()[gpu.id()]->partial_group_gen_powers;

            LDE_launch(gpu, &d_ext_domain[0], &d_domain[0],
                       gen_powers, lg_domain_size, lg_blowup, ext_pow);

            NTT_internal(&d_ext_domain[0], lg_domain_size + lg_blowup,
                         InputOutputOrder::RN, Direction::forward,
                         Type::standard, gpu);

            gpu.DtoH(inout, &d_ext_domain[0], ext_domain_size);
            gpu.sync();
        } catch (const cuda_error& e) {
            gpu.sync();
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
            return RustError{e.code(), e.what()};
#else
            return RustError{e.code()};
#endif
        }

        return RustError{cudaSuccess};
    }

protected:
    static void LDE_launch(stream_t& stream,
                           fr_t* ext_domain_data, fr_t* domain_data,
                           const fr_t (*gen_powers)[WINDOW_SIZE],
                           uint32_t lg_domain_size, uint32_t lg_blowup,
                           bool perform_shift = true, bool ext_pow = false)
    {
        assert(lg_domain_size + lg_blowup <= MAX_LG_DOMAIN_SIZE);
        size_t domain_size = (size_t)1 << lg_domain_size;
        size_t ext_domain_size = domain_size << lg_blowup;

        const cudaDeviceProp& gpu_prop = gpu_props(stream.id());

        // Determine the max power of 2 SM count
        size_t kernel_sms = gpu_prop.multiProcessorCount;
        while (kernel_sms & (kernel_sms - 1))
            kernel_sms -= (kernel_sms & (0 - kernel_sms));

        size_t device_max_threads = kernel_sms * 1024;
        uint32_t num_blocks, block_size;

        if (device_max_threads < domain_size) {
            num_blocks = kernel_sms;
            block_size = 1024;
        } else if (domain_size < 1024) {
            num_blocks = 1;
            block_size = domain_size;
        } else {
            num_blocks = domain_size / 1024;
            block_size = 1024;
        }

        stream.launch_coop(LDE_spread_distribute_powers,
                        {dim3(num_blocks), dim3(block_size),
                         sizeof(fr_t) * block_size},
                        ext_domain_data, domain_data, gen_powers,
                        lg_domain_size, lg_blowup, perform_shift, ext_pow);
    }

public:
    static RustError LDE_aux(const gpu_t& gpu, fr_t* inout,
                             uint32_t lg_domain_size, uint32_t lg_blowup,
                             bool ext_pow = false)
    {
        try {
            size_t domain_size = (size_t)1 << lg_domain_size;
            size_t ext_domain_size = domain_size << lg_blowup;
            // The 2nd to last 'domain_size' chunk will hold the original data
            // The last chunk will get the bit reversed iNTT data
            dev_ptr_t<fr_t> d_inout{ext_domain_size + domain_size, gpu}; // + domain_size for aux buffer
            fr_t* aux_data = &d_inout[ext_domain_size];
            fr_t* domain_data = &d_inout[ext_domain_size - domain_size]; // aligned to the end
            fr_t* ext_domain_data = &d_inout[0];
            gpu.HtoD(domain_data, inout, domain_size);

            NTT_internal(domain_data, lg_domain_size,
                         InputOutputOrder::NR, Direction::inverse,
                         Type::standard, gpu);

            const auto gen_powers =
                NTTParameters::all()[gpu.id()]->partial_group_gen_powers;

            bit_rev(aux_data, domain_data, lg_domain_size, gpu);

            LDE_launch(gpu, ext_domain_data, domain_data, gen_powers,
                       lg_domain_size, lg_blowup, true, ext_pow);

            // NTT - RN
            NTT_internal(ext_domain_data, lg_domain_size + lg_blowup,
                         InputOutputOrder::RN, Direction::forward,
                         Type::standard, gpu);

            gpu.DtoH(inout, ext_domain_data, domain_size << lg_blowup);
            gpu.sync();
        } catch (const cuda_error& e) {
            gpu.sync();
#ifdef TAKE_RESPONSIBILITY_FOR_ERROR_MESSAGE
            return RustError{e.code(), e.what()};
#else
            return RustError{e.code()};
#endif
        }

        return RustError{cudaSuccess};
    }

    // coset_ext_pow is only used when NTT type is coset
    static void Base_dev_ptr(stream_t& stream, fr_t* d_inout,
                             uint32_t lg_domain_size, InputOutputOrder order,
                             Direction direction, Type type,
                             bool coset_ext_pow = false)
    {
        size_t domain_size = (size_t)1 << lg_domain_size;

        NTT_internal(&d_inout[0], lg_domain_size, order, direction, type,
                     stream, coset_ext_pow);
    }

    static void LDE_powers(stream_t& stream, fr_t* d_inout,
                           uint32_t lg_domain_size, bool ext_pow = false)
    {
        LDE_powers(d_inout, false, true, lg_domain_size, 0, stream, ext_pow);
    }

    // If d_out and d_in overlap, d_out is expected to encompass d_in and
    // d_in is expected to be aligned to the end of d_out
    // The input is expected to be in bit-reversed order
    static void LDE_expand(stream_t& stream, fr_t* d_out, fr_t* d_in,
                           uint32_t lg_domain_size, uint32_t lg_blowup)
    {
        LDE_launch(stream, d_out, d_in, NULL, lg_domain_size, lg_blowup, false);
    }
};

#endif
#endif
