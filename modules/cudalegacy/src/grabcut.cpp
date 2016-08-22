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

#include "precomp.hpp"
#include <limits>
#include <stdio.h>
#include <time.h>

using namespace cv;
using namespace cv::cuda;

/////////////////////////////////////////////////////////////////////////
// Debug flags
// #define CHECK_POINT_1
// #define CHECK_POINT_2
// #define CHECK_POINT_3
// #define CHECK_POINT_4
#define PROFILE
/////////////////////////////////////////////////////////////////////////

#if !defined (HAVE_CUDA) || defined (CUDA_DISABLER)

void cv::cuda::grabCut(InputArray, OutputArray, Stream&) { throw_no_cuda(); }

#else /* !defined (HAVE_CUDA) */

////////////////////////////////////////////////////////////////////////
// calcHist
class GMM
{
    public:
        static const int componentsCount = 5;

        GMM( Mat& _model );
        float operator()( const Vec3f color ) const;
        float operator()( int ci, const Vec3f color ) const;
        int whichComponent( const Vec3f color ) const;

        void initLearning();
        void addSample( int ci, const Vec3f color );
        void endLearning();

        void dump(); // debug

        //    private:
        void calcInverseCovAndDeterm( int ci );
        Mat model;
        float* coefs;
        float* mean;
        float* cov;

        float inverseCovs[componentsCount][3][3];
        float covDeterms[componentsCount];

        float sums[componentsCount][3];
        float prods[componentsCount][3][3];
        int sampleCounts[componentsCount];
        int totalSampleCount;
};

GMM::GMM( Mat& _model )
{
    const int modelSize = 3 + 9 + 1;
    if( _model.empty() )
    {
        _model.create( 1, modelSize*componentsCount, CV_32FC1 );
        _model.setTo(Scalar(0));
    }
    else if( (_model.type() != CV_32FC1) || (_model.rows != 1) || (_model.cols != modelSize*componentsCount) )
        CV_Error( CV_StsBadArg, "_model must have CV_32FC1 type, rows == 1 and cols == 13*componentsCount" );

    model = _model;

    coefs = model.ptr<float>(0);
    mean = coefs + componentsCount;
    cov = mean + 3*componentsCount;

    for( int ci = 0; ci < componentsCount; ci++ )
        if( coefs[ci] > 0 )
            calcInverseCovAndDeterm( ci );
}

float GMM::operator()( const Vec3f color ) const
{
    float res = 0;
    for( int ci = 0; ci < componentsCount; ci++ )
        res += coefs[ci] * (*this)(ci, color );
    return res;
}

float GMM::operator()( int ci, const Vec3f color ) const
{
    float res = 0;
    if( coefs[ci] > 0 )
    {
        CV_Assert( covDeterms[ci] > std::numeric_limits<float>::epsilon() );
        Vec3f diff = color;
        float* m = mean + 3*ci;
        diff[0] -= m[0]; diff[1] -= m[1]; diff[2] -= m[2];
        float mult = diff[0]*(diff[0]*inverseCovs[ci][0][0] + diff[1]*inverseCovs[ci][1][0] + diff[2]*inverseCovs[ci][2][0])
            + diff[1]*(diff[0]*inverseCovs[ci][0][1] + diff[1]*inverseCovs[ci][1][1] + diff[2]*inverseCovs[ci][2][1])
            + diff[2]*(diff[0]*inverseCovs[ci][0][2] + diff[1]*inverseCovs[ci][1][2] + diff[2]*inverseCovs[ci][2][2]);
        res = 1.0f/sqrt(covDeterms[ci]) * exp(-0.5f*mult);
    }
    return res;
}

int GMM::whichComponent( const Vec3f color ) const
{
    int k = 0;
    float max = 0;

    for( int ci = 0; ci < componentsCount; ci++ )
    {
        float p = (*this)( ci, color );
        if( p > max )
        {
            k = ci;
            max = p;
        }
    }
    return k;
}

void GMM::dump() {
    for( int ci = 0; ci < componentsCount; ci++)
    {
        printf("%f %f %f\n", sums[ci][0], sums[ci][1], sums[ci][2]);
        printf("%f %f %f\n", prods[ci][0][0], prods[ci][0][1], prods[ci][0][2]);
        printf("%f %f %f\n", prods[ci][1][0], prods[ci][1][1], prods[ci][1][2]);
        printf("%f %f %f\n", prods[ci][2][0], prods[ci][2][1], prods[ci][2][2]);
        printf("%d\n", sampleCounts[ci]);
        printf("-----------------------------------------------------------------\n");
    }
    printf("%d\n", totalSampleCount);
}

void GMM::initLearning()
{
    for( int ci = 0; ci < componentsCount; ci++)
    {
        sums[ci][0] = sums[ci][1] = sums[ci][2] = 0;
        prods[ci][0][0] = prods[ci][0][1] = prods[ci][0][2] = 0;
        prods[ci][1][0] = prods[ci][1][1] = prods[ci][1][2] = 0;
        prods[ci][2][0] = prods[ci][2][1] = prods[ci][2][2] = 0;
        sampleCounts[ci] = 0;
    }
    totalSampleCount = 0;

}

void GMM::addSample( int ci, const Vec3f color )
{
    sums[ci][0] += color[0]; sums[ci][1] += color[1]; sums[ci][2] += color[2];
    prods[ci][0][0] += color[0]*color[0]; prods[ci][0][1] += color[0]*color[1]; prods[ci][0][2] += color[0]*color[2];
    prods[ci][1][0] += color[1]*color[0]; prods[ci][1][1] += color[1]*color[1]; prods[ci][1][2] += color[1]*color[2];
    prods[ci][2][0] += color[2]*color[0]; prods[ci][2][1] += color[2]*color[1]; prods[ci][2][2] += color[2]*color[2];
    sampleCounts[ci]++;
    totalSampleCount++;
}

void GMM::endLearning()
{
    const float variance = 0.01;
    for( int ci = 0; ci < componentsCount; ci++ )
    {
        int n = sampleCounts[ci];
        if( n == 0 )
            coefs[ci] = 0;
        else
        {
            coefs[ci] = (float)n/totalSampleCount;

            float* m = mean + 3*ci;
            m[0] = sums[ci][0]/n; m[1] = sums[ci][1]/n; m[2] = sums[ci][2]/n;

            float* c = cov + 9*ci;
            c[0] = prods[ci][0][0]/n - m[0]*m[0]; c[1] = prods[ci][0][1]/n - m[0]*m[1]; c[2] = prods[ci][0][2]/n - m[0]*m[2];
            c[3] = prods[ci][1][0]/n - m[1]*m[0]; c[4] = prods[ci][1][1]/n - m[1]*m[1]; c[5] = prods[ci][1][2]/n - m[1]*m[2];
            c[6] = prods[ci][2][0]/n - m[2]*m[0]; c[7] = prods[ci][2][1]/n - m[2]*m[1]; c[8] = prods[ci][2][2]/n - m[2]*m[2];

            float dtrm = c[0]*(c[4]*c[8]-c[5]*c[7]) - c[1]*(c[3]*c[8]-c[5]*c[6]) + c[2]*(c[3]*c[7]-c[4]*c[6]);
            if( dtrm <= std::numeric_limits<float>::epsilon() )
            {
                // Adds the white noise to avoid singular covariance matrix.
                c[0] += variance;
                c[4] += variance;
                c[8] += variance;
            }

            calcInverseCovAndDeterm(ci);
        }
    }
}

void GMM::calcInverseCovAndDeterm( int ci )
{
    if( coefs[ci] > 0 )
    {
        float *c = cov + 9*ci;
        float dtrm =
            covDeterms[ci] = c[0]*(c[4]*c[8]-c[5]*c[7]) - c[1]*(c[3]*c[8]-c[5]*c[6]) + c[2]*(c[3]*c[7]-c[4]*c[6]);

        CV_Assert( dtrm > std::numeric_limits<float>::epsilon() );
        inverseCovs[ci][0][0] =  (c[4]*c[8] - c[5]*c[7]) / dtrm;
        inverseCovs[ci][1][0] = -(c[3]*c[8] - c[5]*c[6]) / dtrm;
        inverseCovs[ci][2][0] =  (c[3]*c[7] - c[4]*c[6]) / dtrm;
        inverseCovs[ci][0][1] = -(c[1]*c[8] - c[2]*c[7]) / dtrm;
        inverseCovs[ci][1][1] =  (c[0]*c[8] - c[2]*c[6]) / dtrm;
        inverseCovs[ci][2][1] = -(c[0]*c[7] - c[1]*c[6]) / dtrm;
        inverseCovs[ci][0][2] =  (c[1]*c[5] - c[2]*c[4]) / dtrm;
        inverseCovs[ci][1][2] = -(c[0]*c[5] - c[2]*c[3]) / dtrm;
        inverseCovs[ci][2][2] =  (c[0]*c[4] - c[1]*c[3]) / dtrm;
    }
}

/*
  Calculate beta - parameter of GrabCut algorithm.
  beta = 1/(2*avg(sqr(||color[i] - color[j]||)))
*/
static float calcBeta( const Mat& img )
{
    float beta = 0;
    for( int y = 0; y < img.rows; y++ )
    {
        for( int x = 0; x < img.cols; x++ )
        {
            Vec3f color = img.at<Vec3b>(y,x);
            if( x>0 ) // left
            {
                Vec3f diff = color - (Vec3f)img.at<Vec3b>(y,x-1);
                beta += diff.dot(diff);
            }
            if( y>0 && x>0 ) // upleft
            {
                Vec3f diff = color - (Vec3f)img.at<Vec3b>(y-1,x-1);
                beta += diff.dot(diff);
            }
            if( y>0 ) // up
            {
                Vec3f diff = color - (Vec3f)img.at<Vec3b>(y-1,x);
                beta += diff.dot(diff);
            }
            if( y>0 && x<img.cols-1) // upright
            {
                Vec3f diff = color - (Vec3f)img.at<Vec3b>(y-1,x+1);
                beta += diff.dot(diff);
            }
        }
    }
    if( beta <= std::numeric_limits<float>::epsilon() )
        beta = 0;
    else
        beta = 1.f / (2 * beta/(4*img.cols*img.rows - 3*img.cols - 3*img.rows + 2) );

    return beta;
}

/*
   Check size, type and element values of mask matrix.
   */

static void checkMask( const Mat& img, const Mat& mask )
{
    if( mask.empty() )
        CV_Error( CV_StsBadArg, "mask is empty" );
    if( mask.type() != CV_8UC1 )
        CV_Error( CV_StsBadArg, "mask must have CV_8UC1 type" );
    if( mask.cols != img.cols || mask.rows != img.rows )
        CV_Error( CV_StsBadArg, "mask must have as many rows and cols as img" );
    for( int y = 0; y < mask.rows; y++ )
    {
        for( int x = 0; x < mask.cols; x++ )
        {
            uchar val = mask.at<uchar>(y,x);
            if( val!=GC_BGD && val!=GC_FGD && val!=GC_PR_BGD && val!=GC_PR_FGD )
                CV_Error( CV_StsBadArg, "mask element value must be equel"
                        "GC_BGD or GC_FGD or GC_PR_BGD or GC_PR_FGD" );
        }
    }
}

/*
   Initialize mask using rectangular.
   */

static void initMaskWithRect( Mat& mask, Size imgSize, Rect rect )
{
    mask.create( imgSize, CV_8UC1 );
    mask.setTo( GC_BGD );

    rect.x = std::max(0, rect.x);
    rect.y = std::max(0, rect.y);
    rect.width = std::min(rect.width, imgSize.width-rect.x);
    rect.height = std::min(rect.height, imgSize.height-rect.y);

    (mask(rect)).setTo( Scalar(GC_PR_FGD) );
}

/*
   Initialize GMM background and foreground models using kmeans algorithm.
   */

static void initGMMs( const Mat& img, const Mat& mask, GMM& bgdGMM, GMM& fgdGMM )
{
    const int kMeansItCount = 10;
    const int kMeansType = KMEANS_PP_CENTERS;

    Mat bgdLabels, fgdLabels;
    std::vector<Vec3f> bgdSamples, fgdSamples;
    Point p;
    clock_t t = clock();
    for( p.y = 0; p.y < img.rows; p.y++ )
    {
        for( p.x = 0; p.x < img.cols; p.x++ )
        {
            if( mask.at<uchar>(p) == GC_BGD || mask.at<uchar>(p) == GC_PR_BGD )
                bgdSamples.push_back( (Vec3f)img.at<Vec3b>(p) );
            else // GC_FGD | GC_PR_FGD
                fgdSamples.push_back( (Vec3f)img.at<Vec3b>(p) );
        }
    }
    clock_t diff = clock() - t;
    printf("initGMMs 1 = %f second\n", (float)diff / CLOCKS_PER_SEC);
    t = clock();
    CV_Assert( !bgdSamples.empty() && !fgdSamples.empty() );
    Mat _bgdSamples( (int)bgdSamples.size(), 3, CV_32FC1, &bgdSamples[0][0] );
    kmeans( _bgdSamples, GMM::componentsCount, bgdLabels,
            TermCriteria( CV_TERMCRIT_ITER, kMeansItCount, 0.0), 0, kMeansType );
    Mat _fgdSamples( (int)fgdSamples.size(), 3, CV_32FC1, &fgdSamples[0][0] );
    kmeans( _fgdSamples, GMM::componentsCount, fgdLabels,
            TermCriteria( CV_TERMCRIT_ITER, kMeansItCount, 0.0), 0, kMeansType );
    diff = clock() - t;
    printf("initGMMs 2 = %f second\n", (float)diff / CLOCKS_PER_SEC);

    bgdGMM.initLearning();
    for( int i = 0; i < (int)bgdSamples.size(); i++ )
        bgdGMM.addSample( bgdLabels.at<int>(i,0), bgdSamples[i] );
    bgdGMM.endLearning();

    fgdGMM.initLearning();
    for( int i = 0; i < (int)fgdSamples.size(); i++ )
        fgdGMM.addSample( fgdLabels.at<int>(i,0), fgdSamples[i] );
    fgdGMM.endLearning();
}

/*
  Assign GMMs components for each pixel.
*/
static void assignGMMsComponents( const Mat& img, const Mat& mask, const GMM& bgdGMM, const GMM& fgdGMM, Mat& compIdxs )
{
    Point p;
    for( p.y = 0; p.y < img.rows; p.y++ )
    {
        for( p.x = 0; p.x < img.cols; p.x++ )
        {
            Vec3d color = img.at<Vec3b>(p);
            compIdxs.at<int>(p) = mask.at<uchar>(p) == GC_BGD || mask.at<uchar>(p) == GC_PR_BGD ?
                bgdGMM.whichComponent(color) : fgdGMM.whichComponent(color);
        }
    }
}

/*
  Learn GMMs parameters.
*/
static void learnGMMs( const Mat& img, const Mat& mask, const Mat& compIdxs, GMM& bgdGMM, GMM& fgdGMM )
{
    bgdGMM.initLearning();
    fgdGMM.initLearning();
    Point p;
    for( int ci = 0; ci < GMM::componentsCount; ci++ )
    {
        for( p.y = 0; p.y < img.rows; p.y++ )
        {
            for( p.x = 0; p.x < img.cols; p.x++ )
            {
                if( compIdxs.at<int>(p) == ci )
                {
                    if( mask.at<uchar>(p) == GC_BGD || mask.at<uchar>(p) == GC_PR_BGD )
                        bgdGMM.addSample( ci, img.at<Vec3b>(p) );
                    else
                        fgdGMM.addSample( ci, img.at<Vec3b>(p) );
                }
            }
        }
    }
    bgdGMM.endLearning();
    fgdGMM.endLearning();
}

/*
  Construct GCGraph
*/
static void constructGCGraph( const Mat& img, const Mat& mask, const GMM& bgdGMM, const GMM& fgdGMM, double lambda, Mat& terminals)
{
    Point p;
    for( p.y = 0; p.y < img.rows; p.y++ )
    {
        for( p.x = 0; p.x < img.cols; p.x++)
        {
            Vec3b color = img.at<Vec3b>(p);

            // set t-weights
            float fromSource, toSink;
            if( mask.at<uchar>(p) == GC_PR_BGD || mask.at<uchar>(p) == GC_PR_FGD )
            {
                fromSource = -log( bgdGMM(color) );
                toSink = -log( fgdGMM(color) );
            }
            else if( mask.at<uchar>(p) == GC_BGD )
            {
                fromSource = 0;
                toSink = lambda;
            }
            else // GC_FGD
            {
                fromSource = lambda;
                toSink = 0;
            }
            terminals.at<float>(p) = fromSource - toSink;
        }
    }
}


/*
  Estimate segmentation using MaxFlow algorithm
*/
static void estimateSegmentation( Mat& labels, Mat& mask )
{
    Point p;
    for( p.y = 0; p.y < mask.rows; p.y++ )
    {
        for( p.x = 0; p.x < mask.cols; p.x++ )
        {
            if( mask.at<uchar>(p) == GC_PR_BGD || mask.at<uchar>(p) == GC_PR_FGD )
            {
                if(labels.at<uchar>(p))
                    mask.at<uchar>(p) = GC_PR_FGD;
                else
                    mask.at<uchar>(p) = GC_PR_BGD;
            }
        }
    }
}

namespace gc
{
    //float calcBeta(uchar* data, int rows, int cols, int step);
    void calcNWeights(float *d_pLeft, float *d_pRight, float *d_pTop, float *d_pLeftTop, 
                      float *d_pRightTop, float *d_pBottom, float *d_pBottomLeft, float *d_pBottomRight, 
                      uchar* data, int rows, int cols, int step, int stepNbr, float beta, float gamma);
}

void cv::cuda::grabCut( InputArray _img, InputOutputArray _mask, Rect rect,
        InputOutputArray _bgdModel, InputOutputArray _fgdModel,
        int iterCount, int mode ,Stream &stream )
{
#ifdef PROFILE
    clock_t t, diff;
    clock_t t_t, t_diff;
    t_t = clock();
#endif
    Mat img = _img.getMat();
    Mat& mask = _mask.getMatRef();
    Mat& bgdModel = _bgdModel.getMatRef();
    Mat& fgdModel = _fgdModel.getMatRef();

    if( img.empty() )
        CV_Error( CV_StsBadArg, "image is empty" );
    if( img.type() != CV_8UC3 )
        CV_Error( CV_StsBadArg, "image must have CV_8UC3 type" );

#ifdef PROFILE
    printf("image size = %d x %d\n", img.rows, img.cols);
#endif
    GMM bgdGMM( bgdModel ), fgdGMM( fgdModel );
    Mat compIdxs( img.size(), CV_32SC1 );

    if( mode == GC_INIT_WITH_RECT || mode == GC_INIT_WITH_MASK )
    {

        if( mode == GC_INIT_WITH_RECT )
            initMaskWithRect( mask, img.size(), rect );
        else // flag == GC_INIT_WITH_MASK
            checkMask( img, mask );
#ifdef PROFILE
        t = clock();
#endif
        initGMMs( img, mask, bgdGMM, fgdGMM );
#ifdef PROFILE
        diff = clock() - t;
        printf("initGMMs = %f second\n", (float)diff / CLOCKS_PER_SEC);
#endif
    }

    if( iterCount <= 0)
        return;

    if( mode == GC_EVAL )
        checkMask( img, mask );

    const float gamma = 50;
    const float lambda = 9*gamma;

    Mat terminals(img.size(), CV_32FC1);

#ifdef PROFILE
    t = clock();
#endif
    const float beta = calcBeta( img );
#ifdef PROFILE
    diff = clock() - t;
    printf("calcBeta = %f second\n", (float)diff / CLOCKS_PER_SEC);
#endif
    
#ifdef CHECK_POINT_1
    printf("GPU beta = %f\n", beta);
#endif

#ifdef PROFILE
    t = clock();
#endif
    Size src_size = img.size();
    GpuMat d_img(img),
           d_leftTransp(Size(src_size.height, src_size.width), CV_32FC1),
           d_rightTransp(Size(src_size.height, src_size.width), CV_32FC1),
           d_top(img.size(), CV_32FC1),
           d_topLeft(img.size(), CV_32FC1),
           d_topRight(img.size(), CV_32FC1),
           d_bottom(img.size(), CV_32FC1),
           d_bottomLeft(img.size(), CV_32FC1),
           d_bottomRight(img.size(), CV_32FC1),
           d_buf(img.size(), CV_8U),
           d_labels(img.size(), CV_8U);
#ifdef PROFILE
    diff = clock() - t;
    printf("malloc & memcpy = %f second\n", (float)diff / CLOCKS_PER_SEC);
#endif
    
#ifdef PROFILE
    t = clock();
#endif
    gc::calcNWeights((float*)d_leftTransp.data, (float*)d_rightTransp.data, (float*)d_top.data, (float*)d_topLeft.data, 
                    (float*)d_topRight.data, (float*)d_bottom.data, (float*)d_bottomLeft.data, (float*)d_bottomRight.data, 
                    d_img.data, d_img.rows, d_img.cols, d_img.step, d_top.step, beta, gamma);
#ifdef PROFILE
    diff = clock() - t;
    printf("calcNWeights = %f second\n", (float)diff / CLOCKS_PER_SEC);
#endif

#ifdef CHECK_POINT_2
    Mat leftW;
    d_pLeft.download(leftW);
    for(int y = 0; y < leftW.rows; y++) {
        for(int x = 0; x < leftW.cols; x++) {
            printf("%f\n", leftW.at<float>(y,x));
        }
    }
#endif

    for(int i = 0; i < iterCount; i++) {
        //GCGraph<float> graph;
#ifdef PROFILE
        t = clock();
#endif
        assignGMMsComponents( img, mask, bgdGMM, fgdGMM, compIdxs );
        learnGMMs( img, mask, compIdxs, bgdGMM, fgdGMM );
#ifdef PROFILE
        diff = clock() - t;
        printf("GMM = %f second\n", (float)diff / CLOCKS_PER_SEC);
#endif

#ifdef CHECK_POINT_3
        bgdGMM.dump();
#endif

#ifdef PROFILE
        t = clock();
#endif
        constructGCGraph(img, mask, bgdGMM, fgdGMM, lambda, terminals);

#ifdef PROFILE
        diff = clock() - t;
        printf("constructGCGraph = %f second\n", (float)diff / CLOCKS_PER_SEC);
#endif

#ifdef PROFILE
        t = clock();
#endif
        GpuMat d_terminals(terminals);
        // Four-connected
        cuda::graphcut(d_terminals, d_leftTransp, d_rightTransp, d_top, d_bottom, d_labels, d_buf);
        // Eight-connected
        //cuda::graphcut(d_terminals, d_leftTransp, d_rightTransp, d_top, d_topLeft, d_topRight, d_bottom, d_bottomLeft, d_bottomRight, d_labels, d_buf);
#ifdef PROFILE
        diff = clock() - t;
        printf("GCgraph maxflow = %f second\n", (float)diff / CLOCKS_PER_SEC);
#endif
        //estimateSegmentation( graph, mask );

#ifdef CHECK_POINT_4
        Mat labels;
        d_labels.download(labels);
        for(int y = 0; y < labels.rows; y++) {
            for(int x = 0; x < labels.cols; x++) {
                printf("%d\n", labels.at<uchar>(y,x));
            }
        }
#endif
    }
#ifdef PROFILE
    t_diff = clock() - t_t;
    printf("total = %f second\n", (float)t_diff / CLOCKS_PER_SEC);
#endif
}

#endif /* !defined (HAVE_CUDA) */
