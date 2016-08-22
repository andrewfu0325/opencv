/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/

#if !defined CUDA_DISABLER

#include "stdio.h"
#include "opencv2/core/cuda/common.hpp"
#include "opencv2/core/cuda/functional.hpp"
#include "opencv2/core/cuda/emulation.hpp"
#include "opencv2/core/cuda/transform.hpp"
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <limits>

using namespace cv::cuda;
using namespace cv::cuda::device;


namespace gc
{
    __global__ void calcBetaKernel(uchar* data, float *diff, int rows, int cols, int step) {

        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

        if(x < cols && y < rows) {
            float b = (float)data[(y * step) + x * 3];
            float g = (float)data[(y * step) + x * 3 + 1];
            float r = (float)data[(y * step) + x * 3 + 2];
            if( x>0 ) // left
            {
                float diff_b = b - data[(y * step) + (x - 1) * 3];
                float diff_g = g - data[(y * step) + (x - 1) * 3 + 1];
                float diff_r = r - data[(y * step) + (x - 1) * 3 + 2];
                diff[0 * rows * cols + y * cols + x] = diff_b * diff_b + diff_g * diff_g + diff_r * diff_r;
            }else
                diff[0 * rows * cols + y * cols + x] = 0;
            if( y>0 && x>0 ) // upleft
            {
                float diff_b = b - data[((y - 1) * step) + (x - 1) * 3];
                float diff_g = g - data[((y - 1) * step) + (x - 1) * 3 + 1];
                float diff_r = r - data[((y - 1) * step) + (x - 1) * 3 + 2];
                diff[1 * rows * cols + y * cols + x] = diff_b * diff_b + diff_g * diff_g + diff_r * diff_r;
            }else
                diff[1 * rows * cols + y * cols + x] = 0;
            if( y>0 ) // up
            {
                float diff_b = b - data[((y - 1) * step) + x * 3];
                float diff_g = g - data[((y - 1) * step) + x * 3 + 1];
                float diff_r = r - data[((y - 1) * step) + x * 3 + 2];
                diff[2 * rows * cols + y * cols + x] = diff_b * diff_b + diff_g * diff_g + diff_r * diff_r;
            }else
                diff[2 * rows * cols + y * cols + x] = 0;
            if( y>0 && x<cols-1) // upright
            {
                float diff_b = b - data[((y - 1) * step) + (x + 1) * 3];
                float diff_g = g - data[((y - 1) * step) + (x + 1) * 3 + 1];
                float diff_r = r - data[((y - 1) * step) + (x + 1) * 3 + 2];
                diff[3 * rows * cols + y * cols + x] = diff_b * diff_b + diff_g * diff_g + diff_r * diff_r;
            }else
                diff[3 * rows * cols + y * cols + x] = 0;
        }
    }

    float calcBeta(uchar* data, int rows, int cols, int step) {
        float beta = 0;
        float* diff;
        cudaMalloc(&diff, 4 * cols * rows * sizeof(float));

        dim3 grid((cols - 1) / 16 + 1, (rows - 1) / 16 + 1);
        dim3 block(16, 16);

        calcBetaKernel <<< grid, block >>> (data, diff, rows, cols, step);
        thrust::device_ptr<float> diff_ptr(diff);
        beta = thrust::reduce(diff_ptr, diff_ptr + 4 * rows * cols);
        cudaFree(diff);

        if( beta <= std::numeric_limits<float>::epsilon() )
            beta = 0;
        else
            beta = 1.f / (2 * beta/(4*cols*rows - 3*cols - 3*rows + 2) );
        return beta;
    }
}

namespace gc
{
    __global__ void calcNWeightsKernel(float* pLeftTransp, float* pRightTransp, float* pTop, float* pTopLeft, float* pTopRight, float* pBottom, float* pBottomLeft, float* pBottomRight, uchar* data, int rows, int cols, int step, int stepNbr, float beta, float gamma, float gammaDivSqrt2) {

        const int x = blockIdx.x * blockDim.x + threadIdx.x;
        const int y = blockIdx.y * blockDim.y + threadIdx.y;

        if(x < cols && y < rows) {

            float b = (float)data[(y * step) + x * 3];
            float g = (float)data[(y * step) + x * 3 + 1];
            float r = (float)data[(y * step) + x * 3 + 2];

            if( x-1>=0 ) // left
            {
                float diff_b = b - data[(y * step) + (x - 1) * 3];
                float diff_g = g - data[(y * step) + (x - 1) * 3 + 1];
                float diff_r = r - data[(y * step) + (x - 1) * 3 + 2];
                pLeftTransp[x * stepNbr / 4 + y] = gamma * expf(-beta * (diff_b * diff_b + diff_g * diff_g + diff_r * diff_r));
            }
            else
                pLeftTransp[x * stepNbr / 4 + y] = 0;
            if( x+1<cols ) // right
            {
                float diff_b = b - data[(y * step) + (x + 1) * 3];
                float diff_g = g - data[(y * step) + (x + 1) * 3 + 1];
                float diff_r = r - data[(y * step) + (x + 1) * 3 + 2];
                pRightTransp[x * stepNbr / 4 + y] = gamma * expf(-beta * (diff_b * diff_b + diff_g * diff_g + diff_r * diff_r));
            }
            else
                pRightTransp[x * stepNbr / 4 + y] = 0;
            if( y-1>=0 ) // top
            {
                float diff_b = b - data[((y - 1) * step) + x * 3];
                float diff_g = g - data[((y - 1) * step) + x * 3 + 1];
                float diff_r = r - data[((y - 1) * step) + x * 3 + 2];
                pTop[y * stepNbr / 4 + x] = gamma * expf(-beta * (diff_b * diff_b + diff_g * diff_g + diff_r * diff_r));
            }
            else
                pTop[y * stepNbr / 4 + x] = 0;
            if( x-1>=0 && y-1>=0 ) // topleft
            {
                float diff_b = b - data[((y - 1) * step) + (x - 1) * 3];
                float diff_g = g - data[((y - 1) * step) + (x - 1) * 3 + 1];
                float diff_r = r - data[((y - 1) * step) + (x - 1) * 3 + 2];
                pTopLeft[y * stepNbr / 4 + x] = gammaDivSqrt2 * expf(-beta * (diff_b * diff_b + diff_g * diff_g + diff_r * diff_r));
            }
            else
                pTopLeft[y * stepNbr / 4 + x] = 0;
            if( x+1<cols && y-1>=0 ) // topright
            {
                float diff_b = b - data[((y - 1) * step) + (x + 1) * 3];
                float diff_g = g - data[((y - 1) * step) + (x + 1) * 3 + 1];
                float diff_r = r - data[((y - 1) * step) + (x + 1) * 3 + 2];
                pTopRight[y * stepNbr / 4 + x] = gammaDivSqrt2 * expf(-beta * (diff_b * diff_b + diff_g * diff_g + diff_r * diff_r));
            }
            else
                pTopRight[y * stepNbr / 4 + x] = 0;
            if( y+1<rows ) // bottom
            {
                float diff_b = b - data[((y + 1) * step) + x * 3];
                float diff_g = g - data[((y + 1) * step) + x * 3 + 1];
                float diff_r = r - data[((y + 1) * step) + x * 3 + 2];
                pBottom[y * stepNbr / 4 + x] = gamma * expf(-beta * (diff_b * diff_b + diff_g * diff_g + diff_r * diff_r));
            }
            else
                pBottom[y * stepNbr / 4 + x] = 0;
            if( x-1>=0 && y+1<rows ) // bottomleft
            {
                float diff_b = b - data[((y + 1) * step) + (x - 1) * 3];
                float diff_g = g - data[((y + 1) * step) + (x - 1) * 3 + 1];
                float diff_r = r - data[((y + 1) * step) + (x - 1) * 3 + 2];
                pBottomLeft[y * stepNbr / 4 + x] = gammaDivSqrt2 * expf(-beta * (diff_b * diff_b + diff_g * diff_g + diff_r * diff_r));
            }else
                pBottomLeft[y * stepNbr / 4 + x] = 0;
            if( x+1<cols && y+1<rows ) // bottomright
            {
                float diff_b = b - data[((y + 1) * step) + (x + 1) * 3];
                float diff_g = g - data[((y + 1) * step) + (x + 1) * 3 + 1];
                float diff_r = r - data[((y + 1) * step) + (x + 1) * 3 + 2];
                pBottomRight[y * stepNbr / 4 + x] = gammaDivSqrt2 * expf(-beta * (diff_b * diff_b + diff_g * diff_g + diff_r * diff_r));
            }
            else
                pBottomRight[y * stepNbr / 4 + x] = 0;
        }
    }

    void calcNWeights(float* pLeft, float* pRight, float* pTop, float* pTopLeft, float* pTopRight, float* pBottom, float* pBottomLeft, float* pBottomRight, uchar* data, int rows, int cols, int step, int stepNbr, float beta, float gamma) {

        const float gammaDivSqrt2 = gamma / std::sqrt(2.0f);

        dim3 grid((cols - 1) / 16 + 1, (rows - 1) / 16 + 1);
        dim3 block(16, 16);
        calcNWeightsKernel <<< grid, block >>> (pLeft, pRight, pTop, pTopLeft, pTopRight, pBottom, pBottomLeft, pBottomRight, data, rows, cols, step, stepNbr, beta, gamma, gammaDivSqrt2);
        cudaDeviceSynchronize();
    }
}

/////////////////////////////////////////////////////////////////////////

#endif /* CUDA_DISABLER */
