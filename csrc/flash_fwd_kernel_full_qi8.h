/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cute/algorithm/copy.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>

#include "block_info.h"
#include "kernel_traits.h"
#include "utils.h"
#include "softmax.h"
#include "mask.h"
#include "dropout.h"
#include "rotary.h"

namespace flash {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Return_softmax, typename Params>
inline __device__ void compute_attn_1rowblock_full_qi8(const Params &params, const int bidb, const int bidh, const int m_block) {

    using Element = typename Kernel_traits::Element;
    using Element_Int8 = typename Kernel_traits::Element_Int8;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kNWarps = Kernel_traits::kNWarps;
    constexpr int kGmemScaleElemsPerLoad = Kernel_traits::kGmemScaleElemsPerLoad;
    constexpr int kThrsScaleUsedM = Kernel_traits::kThrsScaleUsedM;
    constexpr int kThrsScaleUsedN = Kernel_traits::kThrsScaleUsedN;
    constexpr int kThrsScaleUsedH = Kernel_traits::kThrsScaleUsedH;

    auto seed_offset = at::cuda::philox::unpack(params.philox_args);
    flash::Dropout dropout(std::get<0>(seed_offset), std::get<1>(seed_offset), params.p_dropout_in_uint8_t,
                           bidb, bidh, tidx, params.h);

    // Save seed and offset for backward, before any early exiting. Otherwise the 0-th thread block might
    // exit early and no one saves the rng states.
    if (Is_dropout && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tidx == 0) {
        params.rng_state[0] = std::get<0>(seed_offset);
        params.rng_state[1] = std::get<1>(seed_offset);
    }

    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    const int n_block_min = !Is_local ? 0 : std::max(0, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) / kBlockN);// 0
    int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);
    if (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max,
                               cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
        // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        //     printf("m_block = %d, n_block_max = %d, actual_seqlen_k = %d, actual_seqlen_q = %d, window_size_right = %d\n", 
        //     m_block, n_block_max, binfo.actual_seqlen_k, binfo.actual_seqlen_q, params.window_size_right);
        // }
    }
    // We exit early and write 0 to gO and gLSE. This also covers the case where actual_seqlen_k == 0.
    // Otherwise we might read OOB elements from gK and gV.
    if ((Is_causal || Is_local || !Is_even_MN) && n_block_max <= n_block_min) {
        Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)
                                              + binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)),
                                make_shape(binfo.actual_seqlen_q, params.h, params.d),
                                make_stride(params.o_row_stride, params.o_head_stride, _1{}));
        Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                              make_coord(m_block, 0));  // (kBlockM, kHeadDim)
        Tensor mLSE = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum*>(params.softmax_lse_ptr)),
                                  make_shape(params.b, params.h, params.seqlen_q),
                                  make_stride(params.h * params.seqlen_q, params.seqlen_q, _1{}));
        Tensor gLSE = local_tile(mLSE(bidb, bidh, _), Shape<Int<kBlockM>>{}, make_coord(m_block));
 
        typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
        Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
        Tensor tOrO = make_tensor<Element>(shape(tOgO));
        clear(tOrO);
        // Construct identity layout for sO
        Tensor cO = make_identity_tensor(make_shape(size<0>(gO), size<1>(gO)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
        // Repeat the partitioning with identity layouts
        Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
        // if (!Is_even_K) {
        //     #pragma unroll
        //     for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d; }
        // }
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_O, tOrO, tOgO, tOcO, binfo.actual_seqlen_q - m_block * kBlockM
        );
        #pragma unroll
        for (int m = 0; m < size<1>(tOgO); ++m) {
            const int row = get<0>(tOcO(0, m, 0));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM && get<1>(tOcO(0, m, 0)) == 0) { gLSE(row) = INFINITY; }
        }
        return;
    }
    // if (tidx == 0) {
    //     printf("m_block = %d, n_block_min = %d, n_block_max = %d\n", m_block, n_block_min, n_block_max);
    // }

    // We iterate over the blocks in reverse order. This is because the last block is the only one
    // that needs masking when we read K and V from global memory. Moreover, iterating in reverse
    // might save us 1 register (we just need n_block instead of both n_block and n_block_max).

    const index_t row_offset_p = ((bidb * params.h + bidh) * params.seqlen_q_rounded
        + m_block * kBlockM) * params.seqlen_k_rounded + (n_block_max - 1) * kBlockN;

    Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element_Int8*>(params.q_ptr)
                                          + binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d),
                            make_stride(params.q_row_stride, params.q_head_stride, _1{}));
    Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));  // (kBlockM, kHeadDim)
    Tensor mQScale = make_tensor(make_gmem_ptr(reinterpret_cast<float*>(params.q_scale_ptr)
                                              + binfo.q_scale_offset(params.q_scale_batch_stride, params.q_scale_row_stride, bidb)),
                                  make_shape(binfo.actual_seqlen_q, params.h),
                                  make_stride(params.q_scale_row_stride, _1{}));
    Tensor gQScale = local_tile(mQScale(_, bidh), Shape<Int<kBlockM>>{}, make_coord(m_block)); // (kBlockM)


    Tensor mK = make_tensor(make_gmem_ptr(reinterpret_cast<Element_Int8*>(params.k_ptr)
                                          + binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb)),
                            make_shape(binfo.actual_seqlen_k, params.h_k, params.d),
                            make_stride(params.k_row_stride, params.k_head_stride, _1{}));
    Tensor gK = local_tile(mK(_, bidh / params.h_h_k_ratio, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
                           make_coord(_, 0));  // (kBlockN, kHeadDim, nblocksN)
    Tensor mKScale = make_tensor(make_gmem_ptr(reinterpret_cast<float*>(params.k_scale_ptr)
                                              + binfo.k_scale_offset(params.k_scale_batch_stride, params.k_scale_row_stride, bidb)),
                                  make_shape(binfo.actual_seqlen_k, params.h_k),
                                  make_stride(params.k_scale_row_stride, _1{}));
    Tensor gKScale = local_tile(mKScale(_, bidh / params.h_h_k_ratio), Shape<Int<kBlockN>>{}, make_coord(_)); // (kBlockN, nblocksN)


    Tensor mV = make_tensor(make_gmem_ptr(reinterpret_cast<Element_Int8*>(params.v_ptr)
                                          + binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb)),
                            make_shape(binfo.actual_seqlen_k, params.h_k, params.d),
                            make_stride(params.v_row_stride, params.v_head_stride, _1{}));
    Tensor gV = local_tile(mV(_, bidh / params.h_h_k_ratio, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
                           make_coord(_, 0));  // (kBlockN, kHeadDim, nblocksN)
    Tensor mVScale = make_tensor(make_gmem_ptr(reinterpret_cast<float*>(params.v_scale_ptr)
                                              + binfo.v_scale_offset(params.v_scale_batch_stride, params.v_scale_row_stride, bidb)),
                                  make_shape(binfo.actual_seqlen_v, params.h_k, kHeadDim),
                                  make_stride(params.v_scale_row_stride, params.v_scale_head_stride, _1{}));
    Tensor gVScale = local_tile(mVScale(_, bidh / params.h_h_k_ratio, _), Shape<Int<1>, Int<kHeadDim>>{}, make_coord(_, 0)); // (1, kHeadDim, nblocksN)

    Tensor gP = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.p_ptr) + row_offset_p),
                            Shape<Int<kBlockM>, Int<kBlockN>>{},
                            make_stride(params.seqlen_k_rounded, _1{}));

    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element_Int8 *>(smem_)),
                            typename Kernel_traits::SmemLayoutQ_Int8{});
    Tensor sK = make_tensor(sQ.data() + size(sQ),
                            typename Kernel_traits::SmemLayoutKV_Int8{});
    Tensor sV = make_tensor(sK.data() + size(sK),
                            typename Kernel_traits::SmemLayoutKV_Int8{});
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed_Int8{});
    Tensor sVtNoSwizzle = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle_Int8{});
    Tensor sP = make_tensor(sV.data() + size(sV), typename Kernel_traits::SmemLayoutP_Int8{});

    const int offset1 = (size(sQ) + size(sK) + size(sV) + size(sP)) * sizeof(Element_Int8);
    Tensor sQS = make_tensor(make_smem_ptr(reinterpret_cast<float*>(smem_ + offset1)), typename Kernel_traits::SmemLayoutQScale{});
    Tensor sKS = make_tensor(sQS.data() + size(sQS), typename Kernel_traits::SmemLayoutKVScale{});
    Tensor sVS = make_tensor(sKS.data() + size(sKS), typename Kernel_traits::SmemLayoutVScale{});


    typename Kernel_traits::GmemTiledCopyQKV_Int8 gmem_tiled_copy_QKV_Int8;
    auto gmem_thr_copy_QKV_Int8 = gmem_tiled_copy_QKV_Int8.get_thread_slice(tidx);
    // typename Kernel_traits::GmemTiledCopyQKVScale gmem_tiled_copy_QKVScale;
    // auto gmem_thr_copy_QKVScale = gmem_tiled_copy_QKVScale.get_thread_slice(tidx);

    Tensor tQgQ = gmem_thr_copy_QKV_Int8.partition_S(gQ); // (QCPY, QCPY_M, QCPY_K)
    Tensor tQsQ = gmem_thr_copy_QKV_Int8.partition_D(sQ);
    Tensor tKgK = gmem_thr_copy_QKV_Int8.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K, nblocksN)
    Tensor tKsK = gmem_thr_copy_QKV_Int8.partition_D(sK);
    Tensor tVgV = gmem_thr_copy_QKV_Int8.partition_S(gV);  // (VCPY, VCPY_N, VCPY_K, nblocksN)
    Tensor tVsV = gmem_thr_copy_QKV_Int8.partition_D(sV);

    typename Kernel_traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    typename Kernel_traits::TiledMma_Int8 tiled_mma_Int8;
    auto thr_mma_Int8 = tiled_mma_Int8.get_thread_slice(tidx);

    Tensor tSrQ  = thr_mma_Int8.partition_fragment_A(sQ);                          // (MMA,MMA_M,MMA_K)
    Tensor tSrK  = thr_mma_Int8.partition_fragment_B(sK);                          // (MMA,MMA_N,MMA_K)
    Tensor tOrP  = thr_mma_Int8.partition_fragment_A(sP);                          // (MMA, MMA_M,MMA_N)
    Tensor tOrVt = thr_mma_Int8.partition_fragment_B(sVtNoSwizzle);                // (MMA, MMA_K,MMA_N)

    Tensor tSgS  = thr_mma.partition_C(gP);////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});  // MMA, MMA_M, MMA_K
    Tensor acc_o_int = partition_fragment_C(tiled_mma_Int8, Shape<Int<kBlockM>, Int<kHeadDim>>{});  // MMA, MMA_M, MMA_K
    Tensor acc_o = flash::convert_type_tensor<float>(acc_o_int);

    //
    // Copy Atom retiling
    //

    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom_Int8{}, tiled_mma_Int8);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
    // if (cute::thread0()) {smem_thr_copy_Q.print_all();}
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);
    // if (cute::thread0()) {print(tSsQ.layout()); printf("\n");}

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom_Int8{}, tiled_mma_Int8);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    auto smem_tiled_copy_P = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom_Int8{}, tiled_mma_Int8);
    auto smem_thr_copy_P = smem_tiled_copy_P.get_thread_slice(tidx);
    Tensor tOsP = smem_thr_copy_P.partition_S(sP);

    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed_Int8{}, tiled_mma_Int8);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

    auto smem_tiled_copy_P_B = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomP{}, tiled_mma_Int8);
    auto smem_thr_copy_P_B = smem_tiled_copy_P_B.get_thread_slice(tidx);
    Tensor tPsP = smem_thr_copy_P_B.partition_D(sP);

    //
    // PREDICATES
    //

    // // Allocate predicate tensors for m and n
    // Tensor tQpQ = make_tensor<bool>(make_shape(size<1>(tQsQ), size<2>(tQsQ)), Stride<_1,_0>{});
    // Tensor tKVpKV = make_tensor<bool>(make_shape(size<1>(tKsK), size<2>(tKsK)), Stride<_1,_0>{});

    // Construct identity layout for sQ and sK
    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    // Tensor tScQ = thr_mma.partition_A(cQ);                           // (MMA,MMA_M,MMA_K)
    // if (cute::thread0()) {
    //     print(tScQ.layout()); printf("\n");
    //     for (int i = 0; i < size(tScQ); ++i) {
    //         printf("%d ", get<0>(tScQ(i)));
    //     }
    //     printf("\n");
    //     for (int i = 0; i < size(tScQ); ++i) {
    //         printf("%d ", get<1>(tScQ(i)));
    //     }
    //     printf("\n");
    // }

    // Repeat the partitioning with identity layouts
    Tensor tQcQ = gmem_thr_copy_QKV_Int8.partition_S(cQ);       // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tKcK = gmem_thr_copy_QKV_Int8.partition_S(cKV);
    Tensor tVcV = gmem_thr_copy_QKV_Int8.partition_S(cKV);

    // Prologue

    // We don't need to clear the sQ smem tiles since we'll only write out the valid outputs
    flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV_Int8, tQgQ, tQsQ, tQcQ,
                                       binfo.actual_seqlen_q - m_block * kBlockM);
    // flash::copy_scale<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKVScale, tQSgQS, tQSsQS,
    //                                          kBlockM / 8);
    flash::copy_simple<kGmemScaleElemsPerLoad, kThrsScaleUsedM, Is_even_MN>(gQScale, sQS,
                                                                            binfo.actual_seqlen_q - m_block * kBlockM);
    // if (Kernel_traits::Is_Q_in_regs) { cute::cp_async_fence(); }

    // // if (cute::thread(1, 0)) { print(tQsQ); }
    // // Tensor sQNoSwizzle = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)), typename Kernel_traits::SmemLayoutQNoSwizzle{});
    // // if (cute::thread0()) { print(sQNoSwizzle); }

    // if (Kernel_traits::Share_Q_K_smem) {
    //     flash::cp_async_wait<0>();
    //     __syncthreads();
    //     Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
    //     CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));            // M
    //     cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
    //     __syncthreads();
    // }

    int n_block = n_block_max - 1;
    // We don't need to clear the sK smem tiles since we'll mask out the scores anyway.
    flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV_Int8, tKgK(_, _, _, n_block), tKsK, tKcK,
                                       binfo.actual_seqlen_k - n_block * kBlockN);
    flash::copy_simple<kGmemScaleElemsPerLoad, kThrsScaleUsedN, Is_even_MN>(gKScale(_, n_block),
                                                                            sKS, binfo.actual_seqlen_k - n_block * kBlockN);
    cute::cp_async_fence();
    // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z < 2) { print(tKgK); }
    // __syncthreads();

    // if (Kernel_traits::Is_Q_in_regs && !Kernel_traits::Share_Q_K_smem) {
    //     flash::cp_async_wait<1>();
    //     __syncthreads();
    //     Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
    //     CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));            // M
    //     cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
    // }
    // if(tidx == 0){
    //     printf("End of thr first copy.\n");
    //     print_tensor(sQ);
    //     print_tensor(sK);
    //     print_tensor(sQS);
    //     print_tensor(sKS);
    // }
    clear(acc_o);

    flash::Softmax<2 * size<1>(acc_o)> softmax;

    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Mask<Is_causal, Is_local, Has_alibi> mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q, params.window_size_left, params.window_size_right, alibi_slope);

    // For performance reason, we separate out two kinds of iterations:
    // those that need masking on S, and those that don't.
    // We need masking on S for the very last block when K and V has length not multiple of kBlockN.
    // We also need masking on S if it's causal, for the last ceil_div(kBlockM, kBlockN) blocks.
    // We will have at least 1 "masking" iteration.

    // If not even_N, then seqlen_k might end in the middle of a block. In that case we need to
    // mask 2 blocks (e.g. when kBlockM == kBlockN), not just 1.
    constexpr int n_masking_steps = (!Is_causal && !Is_local)
        ? 1
        : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1);
    #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        Tensor acc_s_int = partition_fragment_C(tiled_mma_Int8, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
        clear(acc_s_int);
        clear(acc_o_int);
        flash::cp_async_wait<0>();
        __syncthreads();

        // Advance gV
        if (masking_step > 0) {
            flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV_Int8, tVgV(_, _, _, n_block), tVsV, tVcV);
        } else {
            // Clear the smem tiles to account for predicated off loads
            flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
                gmem_tiled_copy_QKV_Int8, tVgV(_, _, _, n_block), tVsV, tVcV, binfo.actual_seqlen_k - n_block * kBlockN
            );
        }
        flash::copy_simple<kGmemScaleElemsPerLoad, kThrsScaleUsedH>(gVScale(0, _, n_block), sVS);
        cute::cp_async_fence();

        flash::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
            acc_s_int, tSrQ, tSrK, tSsQ, tSsK, tiled_mma_Int8, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K
        );
        // Convert acc_s_int from int32 to fp32
        Tensor acc_s = flash::convert_type_tensor<float>(acc_s_int);
        mask.template apply_rescale_mask<Is_causal, Is_even_MN>(
            acc_s, acc_s_int, sQS, sKS, n_block * kBlockN, m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4, kNWarps * 16, 
            n_block * kBlockN, m_block * kBlockM
        );
        // if(m_block == 1 && tidx == 0){
        //     // mask.template print_s(
        //     //     acc_s_int, n_block * kBlockN, m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4, kNWarps * 16
        //     // );
        //     print_tensor(sQS);
        //     print_tensor(sKS);
        // }

        flash::cp_async_wait<0>();
        __syncthreads();
        // Advance gK
        if (n_block > n_block_min) {
            flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV_Int8, tKgK(_, _, _, n_block - 1), tKsK, tKcK);
            // flash::copy_scale<true, Is_even_K>(gmem_tiled_copy_QKVScale, tKSgKS(_, _, n_block - 1), tKSsKS, kBlockN / 8);
            flash::copy_simple<kGmemScaleElemsPerLoad, kThrsScaleUsedN>(gKScale(_, n_block - 1), sKS);
            // This cp_async_fence needs to be in the if block, otherwise the synchronization
            // isn't right and we get race conditions.
            cute::cp_async_fence();
        }

        // TODO: when we have key_padding_mask we'll need to Check_inf
        masking_step == 0
            ? softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2)
            : softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2);

        // Convert acc_s from fp32 to int8
        flash::scale_apply_f(acc_s, 127.0f);
        Tensor rP = flash::convert_type<Element_Int8>(acc_s);

        // Write P back to shared memory
        Tensor tPrP = smem_thr_copy_P_B.retile_S(rP);
        cute::copy(smem_tiled_copy_P_B, tPrP, tPsP);
        flash::cp_async_wait<0>();

        flash::gemm(
            acc_o_int, tOrP, tOrVt, tOsP, tOsVt, tiled_mma_Int8, smem_tiled_copy_P, smem_tiled_copy_V,
            smem_thr_copy_P, smem_thr_copy_V
        );
        mask.template apply_rescale_v(acc_o, acc_o_int, sVS);
        // if (cute::thread0()) { print(scores); }

        // This check is at the end of the loop since we always have at least 1 iteration
        if (n_masking_steps > 1 && n_block <= n_block_min) {
            --n_block;
            break;
        }
    }

    // if(tidx == 0){
    //     printf("End of masking calc.\n");
    // }

    // These are the iterations where we don't need masking on S
    for (; n_block >= n_block_min; --n_block) {
        Tensor acc_s_int = partition_fragment_C(tiled_mma_Int8, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
        clear(acc_s_int);
        clear(acc_o_int);
        flash::cp_async_wait<0>();
        __syncthreads();

        flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV_Int8, tVgV(_, _, _, n_block), tVsV, tVcV);
        flash::copy_simple<kGmemScaleElemsPerLoad, kThrsScaleUsedH>(gVScale(0, _, n_block), sVS);
        cute::cp_async_fence();

        flash::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
            acc_s_int, tSrQ, tSrK, tSsQ, tSsK, tiled_mma_Int8, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K
        );
        Tensor acc_s = flash::convert_type_tensor<float>(acc_s_int);
        mask.template apply_rescale_mask</*Causal_mask=*/false>(
            acc_s, acc_s_int, sQS, sKS, n_block * kBlockN, m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4, kNWarps * 16,
            n_block * kBlockN, m_block * kBlockM
        );
        // if(m_block == 1 && tidx == 0){
        //     // mask.template print_s(
        //     //     acc_s_int, n_block * kBlockN, m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4, kNWarps * 16
        //     // );
        //     print_tensor(sQS);
        //     print_tensor(sKS);
        // }

        flash::cp_async_wait<0>();
        __syncthreads();
        // advance gK
        if (n_block > n_block_min) {
            flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV_Int8, tKgK(_, _, _, n_block - 1), tKsK, tKcK);
            // flash::copy_scale<true, Is_even_K>(gmem_tiled_copy_QKVScale, tKSgKS(_, _, n_block - 1), tKSsKS, kBlockN / 8);
            flash::copy_simple<kGmemScaleElemsPerLoad, kThrsScaleUsedN>(gKScale(_, n_block - 1), sKS);
            // This cp_async_fence needs to be in the if block, otherwise the synchronization
            // isn't right and we get race conditions.
            cute::cp_async_fence();
        }

        softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_local>(acc_s, acc_o, params.scale_softmax_log2);

        // Convert acc_s from fp32 to int8
        flash::scale_apply_f(acc_s, 127.0f);
        Tensor rP = flash::convert_type<Element_Int8>(acc_s);

        Tensor tPrP = smem_thr_copy_P_B.retile_S(rP);
        cute::copy(smem_tiled_copy_P_B, tPrP, tPsP);
        flash::cp_async_wait<0>();

        flash::gemm(
            acc_o_int, tOrP, tOrVt, tOsP, tOsVt, tiled_mma_Int8, smem_tiled_copy_P, smem_tiled_copy_V,
            smem_thr_copy_P, smem_thr_copy_V
        );
        mask.template apply_rescale_v(acc_o, acc_o_int, sVS);
    }

    // Epilogue

    Tensor lse = softmax.template normalize_softmax_lse<Is_dropout>(acc_o, params.scale_softmax, params.rp_dropout);

    // Convert acc_o from fp32 to fp16/bf16
    Tensor rO = flash::convert_type<Element>(acc_o);
    Tensor sO = make_tensor(make_smem_ptr(reinterpret_cast<Element*>(smem_)), 
                            typename Kernel_traits::SmemLayoutO{});    // (SMEM_M,SMEM_N)
    // Partition sO to match the accumulator partitioning
    auto smem_tiled_copy_O = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma);
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
    Tensor taccOrO = smem_thr_copy_O.retile_S(rO);        // ((Atom,AtomNum), MMA_M, MMA_N)
    Tensor taccOsO = smem_thr_copy_O.partition_D(sO);     // ((Atom,AtomNum),PIPE_M,PIPE_N)

    // sO has the same size as sQ, so we don't need to sync here.
    // if (Kernel_traits::Share_Q_K_smem) { __syncthreads(); }

    cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

    Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)
                                          + binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)),
                            make_shape(binfo.actual_seqlen_q, params.h, params.d),
                            make_stride(params.o_row_stride, params.o_head_stride, _1{}));
    Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                           make_coord(m_block, 0));  // (kBlockM, kHeadDim)
    Tensor mLSE = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum*>(params.softmax_lse_ptr)),
                              make_shape(params.b, params.h, params.seqlen_q),
                              make_stride(params.h * params.seqlen_q, params.seqlen_q, _1{}));
    Tensor gLSE = local_tile(mLSE(bidb, bidh, _), Shape<Int<kBlockM>>{}, make_coord(m_block));

    typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
    Tensor tOsO = gmem_thr_copy_O.partition_S(sO);        // ((Atom,AtomNum),ATOM_M,ATOM_N)
    Tensor tOgO = gmem_thr_copy_O.partition_D(gO);

    __syncthreads();

    Tensor tOrO = make_tensor<Element>(shape(tOgO));
    cute::copy(gmem_tiled_copy_O, tOsO, tOrO);

    Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor taccOcO = thr_mma.partition_C(caccO);                           // (MMA,MMA_M,MMA_K)
    static_assert(decltype(size<0>(taccOcO))::value == 4);
    // Convert to ((2, 2), MMA_M, MMA_K) then take only the row indices.
    Tensor taccOcO_row = logical_divide(taccOcO, Shape<_2>{})(make_coord(0, _), _, 0);
    CUTE_STATIC_ASSERT_V(size(lse) == size(taccOcO_row));                     // MMA_M
    if (get<1>(taccOcO_row(0)) == 0) {
        #pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            const int row = get<0>(taccOcO_row(mi));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM) { gLSE(row) = lse(mi); }
        }
    }

    // Construct identity layout for sO
    Tensor cO = make_identity_tensor(make_shape(size<0>(sO), size<1>(sO)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    // Repeat the partitioning with identity layouts
    Tensor tOcO = gmem_thr_copy_O.partition_D(cO);                           // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d; }
    }
    // Clear_OOB_K must be false since we don't want to write zeros to gmem
    flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_O, tOrO, tOgO, tOcO, binfo.actual_seqlen_q - m_block * kBlockM
    );
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Return_softmax, typename Params>
inline __device__ void compute_attn_full_qi8(const Params &params) {
    const int m_block = blockIdx.x;
    // The block index for the batch.
    const int bidb = blockIdx.y;
    // The block index for the head.
    const int bidh = blockIdx.z;

    // We want the fwd and bwd to generate the same dropout pattern (RNG), without restricting
    // them to have the same number of threads or have to traverse the attention matrix
    // in the same order.
    // In the Philox RNG, we use the offset to store the batch, head, and the lane id
    // (within a warp). We use the subsequence to store the location of the 16 x 32 blocks within
    // the attention matrix. This way, as long as we have the batch, head, and the location of
    // the 16 x 32 block within the attention matrix, we can generate the exact same dropout pattern.

    flash::compute_attn_1rowblock_full_qi8<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Return_softmax>(params, bidb, bidh, m_block);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

} // namespace flash
