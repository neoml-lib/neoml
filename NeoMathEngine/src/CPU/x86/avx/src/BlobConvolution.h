/* Copyright © 2017-2020 ABBYY Production LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
--------------------------------------------------------------------------------------------------------------*/

#include <array>
#include <vector>
#include <algorithm>
#include <utility>
#include <memory>
#include <cstring>
#include <functional>

#include <NeoMathEngine/NeoMathEngine.h>
#include <JitCommon.h>

namespace NeoML {

using reg64_t = Xbyak::Reg64;

class CBlobConvolutionBase : public CCrtAllocatedObject {
public:
    virtual ~CBlobConvolutionBase() = default;
    virtual void ProcessConvolution( int threadCount, const float* sourceData, const float* filterData, const float* freeTermData, float* resultData ) = 0;
};

template<int FltCnt>
class CBlobConvolution : public CBlobConvolutionBase {
public:
    CBlobConvolution(
        IMathEngine* mathEngine,
        int channelCount, int filterHeight, int filterWidth, int sourceHeight, int sourceWidth,
        int paddingHeight, int paddingWidth, int strideHeight, int strideWidth,
        int dilationHeight, int dilationWidth, int resultHeight, int resultWidth, int resObjCnt );
    ~CBlobConvolution() override = default;

    void ProcessConvolution( int threadCount,
        const float* sourceData, const float* filterData, const float* freeTermData, float* resultData ) override;

private:
    struct CSize {
        int Height;
        int Width;
    };

    static const int WideBatchKernelHeight;
    static const int WideBatchKernelWidth;
    static const int NarrowBatchKernelHeight;
    static const int NarrowBatchKernelWidth;

    // Class wich handles generation and running JIT code for convolution.
    class CJitConvolution : public Xbyak::CodeGenerator {
    public:
        // Init JIT code main routine
        CJitConvolution( CBlobConvolution& bc, int yStepIndex );

        void Run( bool useNarrowProcessing, const float* srcPtr, const float* fltPtr, const float* freeTermPtr, float* resPtr );

        static constexpr unsigned int NumFloatInYmm = 8;
        static constexpr unsigned int SizeOfYmm = NumFloatInYmm * sizeof( float );
    private:
        // Passed to 'Run()' function as arguments
        const reg64_t regUseNarrowProcessing = Param1;
        const reg64_t regSrcPtr = Param2;
        const reg64_t regFltPtr = Param3;
        const reg64_t regFreeTermPtr = Param4;
#ifdef _WIN32
        const reg64_t regResPtr = Xbyak::util::rdi;
#else
        const reg64_t regResPtr = Param5;
#endif

        // Used in kernels:
        const reg64_t regTempSrcPtr = Xbyak::util::r10;
        const reg64_t regTempFltPtr = Xbyak::util::r11;
        const reg64_t retTemp = Xbyak::util::r14;
        const reg64_t regNumSteps = Xbyak::util::r12;
        const reg64_t regChCnt = Xbyak::util::rax;

        void prologue();
        void epilogue();
        void labelAlign( Xbyak::Label& label, int alignment = 16 ) {
            align( alignment );
            L( label );
        }

        void fillBatchProcessingKernel( CBlobConvolution<FltCnt>& bc, bool useNarrowProcessing, size_t windowIndex );
        void fillSingleProcessingKernel( CBlobConvolution<FltCnt>& bc, bool useNarrowProcessing, size_t windowIndex );

        // Initialize result registers with data from freeTerm (if it isn't nullptr)
        void initResRegs( Xbyak::Ymm* res, Xbyak::Ymm* tempRes, size_t KernelHeight, size_t KernelWidth );
        // Flush result registers
        // 'callBeforeFlush' will be called before flushing of result registers. It can be captured labda function.
        void flushResRegs( Xbyak::Ymm* res, size_t KernelHeight, size_t KernelWidth, bool useNarrowProcessing, size_t resNarrowStep );
        void initProcessingMainLoop( CBlobConvolution<FltCnt>& bc, Xbyak::Ymm* res, Xbyak::Ymm* tempRes,
            size_t stepCount, size_t stepSize,
            Xbyak::Label& labelKernel, Xbyak::Label& labelEndOfProcessingFunction,
            size_t windowIndex, bool useNarrowProcessing = false, size_t resNarrowStep = 0,
            std::function<void()>* callBeforeFlush = nullptr );

        // Circular rotate y0, y1 and y2 to the left at 6 floats using 3 additional temporary registers.
        void rotateLeft6( Xbyak::Ymm& y0, Xbyak::Ymm& y1, Xbyak::Ymm& y2,
            Xbyak::Ymm& yt0, Xbyak::Ymm& yt1, Xbyak::Ymm& yt2 );
        // Circular rotate y0 to the left at 2 floats using 1 additional temporary register.
        void rotateLeft2( Xbyak::Ymm& y, Xbyak::Ymm& yt );
        // Circular rotate y to the right at 2
        void rotateRight2( Xbyak::Ymm& dst, Xbyak::Ymm& src );
    };

    IMathEngine* mathEngine;

    const int ChCnt;
    const int FltH;
    const int FltW;
    const int SrcH;
    const int SrcW;
    const int PaddingH;
    const int PaddingW;
    const int StrideH;
    const int StrideW;
    const int DilationH;
    const int DilationW;
    const int ResH;
    const int ResW;
    const int ResObjCnt;
    bool jitIsInited;

    // For some cases we will use FltCnt, rounded up to nearest integer multiple of 8
    static constexpr int FltCntM8 = ( FltCnt + 8 - 1 ) / 8 * 8;
    static constexpr size_t AvxAlignment = 32;

    const float* src;
    const float* flt;
    const float* freeTerm;
    float* res;

    // !!! SrcXStep, SrcYStep and ResLineStride are read from JIT as 8-byte values, hence they must have 8 byte length.
    // Length of one source line.
    const size_t SrcLineStride;
    // Distance in floats between two neighbor pixels in source.
    const size_t SrcXStep;
    const size_t SrcYStep;
    // Distance in floats between two neighbor pixels in window by horizontal.
    const size_t SrcXDilation;
    // Distance in floats between two neighbor pixels in window by horizontal.
    const size_t SrcYDilation;
    // Width of source window in floats
    const size_t SrcXWindowSize;
    const size_t ResLineStride;

    // When we move filter window over the source image we have different combination of intersection this window with
    // source image. Filters window moves left to right and up to bottom, "PixelOffsetResStepsWidth..." - is number
    // of steps which window moved till its intersection with source image changed.We will calculate steps over width and heigh.
    std::vector<int> PixelOffsetResStepsWidthX;
    std::vector<int> PixelOffsetResStepsWidthY;

    // Choose proper pixels in source and filter:
    // 0  1  2
    // 3  4  5
    // 6  7  8 (example for 3x3)
    // Offset is relative to central pixel of source window
    std::vector<std::vector<int>> SrcPixelsOffset;
    // Offset is relative to center pixel of filter window
    std::vector<std::vector<int>>  FltPixelsOffset;
    // In some cases when the width of the image is nearly equals to the width of optimized batch processing window,
    // we may faced to situation ( when dilation is higth ) when no one optimized batch ptocessing can be
    // applied. For such cases we will use optimized batch processing with narrower window but height greater then one.
    const CSize NarrowBatchProcessSize;
    const CSize WideBatchProcessSize;

    std::vector<std::unique_ptr<CJitConvolution>> jitCodes;

    // Initialize NarrowBatchProcessSize and WideBatchProcessSize
    CSize getNarrowBatchProcessSize();
    CSize getWideBatchProcessSize();

    void initJitCodes();

    // Rearrange filter and fill 'Filter' and 'FreeTerm' members.
    const float* rearrangeFilter( const float* filterData, CFloatHandleStackVar& Filter );
    const float* rearrangeFreeTerm( const float* freeTermData, CFloatHandleStackVar& FreeTerm );
    // Function calculates offsets of center of filter window over the source image, where intersection over
    // them is changed. This function helps to calculate further PixelOffsetResStepsWidthX/Y, SrcPixelsOffset and FltPixelsOffset.
    // Src (source), F(filter), D(dilation), S(stride) and P(padding) linear dimention by X or Y axis.
    std::vector<int> getPixelOffsetSrcSteps( int SrcDim, int FDim, int DDim, int SDim, int PDim );

    // Initialize PixelOffsetResStepsX, PixelOffsetResStepsY, SrcPixelsOffset and FltPixelsOffset
    void fillPixelOffset();
};

template<int FltCnt>
const int CBlobConvolution<FltCnt>::NarrowBatchKernelHeight = INT_MAX;

template<int FltCnt>
const int CBlobConvolution<FltCnt>::NarrowBatchKernelWidth = INT_MAX;

class CBlobConvolutionFabric : public CCrtAllocatedObject {
public:
    static bool IsBlobConvolutionAvailable( int FltCnt, int FltH, int FltW );
    static std::unique_ptr<CBlobConvolutionBase> GetProperInstance(
        IMathEngine* mathEngine, int FltCnt,
        int channelCount, int filterHeight, int filterWidth, int sourceHeight, int sourceWidth,
        int paddingHeight, int paddingWidth, int strideHeight, int strideWidth,
        int dilationHeight, int dilationWidth, int resultHeight, int resultWidth, int resObjCnt );
};

} // namespace NeoML

#include <BlobConvolution.inl>
// JIT
#include <BlobConvolution_jit.inl>
#include <BlobConvolution_jit_FltCnt_3.inl>
#include <BlobConvolution_jit_FltCnt_6.inl>
#include <BlobConvolution_jit_FltCnt_8.inl>
#include <BlobConvolution_jit_FltCnt_16.inl>
#include <BlobConvolution_jit_FltCnt_18.inl>
#include <BlobConvolution_jit_FltCnt_24.inl>
#include <BlobConvolution_jit_FltCnt_32.inl>