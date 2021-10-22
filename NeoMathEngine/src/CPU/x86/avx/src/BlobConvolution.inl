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

namespace NeoML {

bool CBlobConvolutionFabric::IsBlobConvolutionAvailable( int FltCnt, int FltH, int FltW )
{
    if( FltH % 2 == 0 || FltW % 2 == 0 ) {
        return false;
    }
    if(
        FltCnt == 32 ||
        FltCnt == 24 ||
        FltCnt == 18 ||
        FltCnt == 16 ||
        FltCnt == 8 ||
        FltCnt == 6 ||
        FltCnt == 3 ) {
        return true;
    }
    return false;
}

std::unique_ptr<CBlobConvolutionBase> CBlobConvolutionFabric::GetProperInstance(
    IMathEngine* mathEngine, int filterCount,
    int channelCount, int filterHeight, int filterWidth, int sourceHeight, int sourceWidth,
    int paddingHeight, int paddingWidth, int strideHeight, int strideWidth,
    int dilationHeight, int dilationWidth, int resultHeight, int resultWidth, int resObjCnt )
{
    switch( filterCount ) {
    case 32:
        return std::unique_ptr<CBlobConvolutionBase>(
            new CBlobConvolution<32>(
                mathEngine, channelCount, filterHeight, filterWidth, sourceHeight, sourceWidth,
                paddingHeight, paddingWidth, strideHeight, strideWidth,
                dilationHeight, dilationWidth, resultHeight, resultWidth, resObjCnt ) );
    case 24:
        return std::unique_ptr<CBlobConvolutionBase>(
            new CBlobConvolution<24>(
                mathEngine, channelCount, filterHeight, filterWidth, sourceHeight, sourceWidth,
                paddingHeight, paddingWidth, strideHeight, strideWidth,
                dilationHeight, dilationWidth, resultHeight, resultWidth, resObjCnt ) );
    case 18:
        return std::unique_ptr<CBlobConvolutionBase>(
            new CBlobConvolution<18>(
                mathEngine, channelCount, filterHeight, filterWidth, sourceHeight, sourceWidth,
                paddingHeight, paddingWidth, strideHeight, strideWidth,
                dilationHeight, dilationWidth, resultHeight, resultWidth, resObjCnt ) );
    case 16:
        return std::unique_ptr<CBlobConvolutionBase>(
            new CBlobConvolution<16>(
                mathEngine, channelCount, filterHeight, filterWidth, sourceHeight, sourceWidth,
                paddingHeight, paddingWidth, strideHeight, strideWidth,
                dilationHeight, dilationWidth, resultHeight, resultWidth, resObjCnt ) );
    case 8:
        return std::unique_ptr<CBlobConvolutionBase>(
            new CBlobConvolution<8>(
                mathEngine, channelCount, filterHeight, filterWidth, sourceHeight, sourceWidth,
                paddingHeight, paddingWidth, strideHeight, strideWidth,
                dilationHeight, dilationWidth, resultHeight, resultWidth, resObjCnt ) );
    case 6:
        return std::unique_ptr<CBlobConvolutionBase>(
            new CBlobConvolution<6>(
                mathEngine, channelCount, filterHeight, filterWidth, sourceHeight, sourceWidth,
                paddingHeight, paddingWidth, strideHeight, strideWidth,
                dilationHeight, dilationWidth, resultHeight, resultWidth, resObjCnt ) );
    case 3:
        return std::unique_ptr<CBlobConvolutionBase>(
            new CBlobConvolution<3>(
                mathEngine, channelCount, filterHeight, filterWidth, sourceHeight, sourceWidth,
                paddingHeight, paddingWidth, strideHeight, strideWidth,
                dilationHeight, dilationWidth, resultHeight, resultWidth, resObjCnt ) );
    default:
        return nullptr;
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

template<int FltCnt>
CBlobConvolution<FltCnt>::CBlobConvolution(
    IMathEngine* _mathEngine, int channelCount, int filterHeight, int filterWidth,
    int sourceHeight, int sourceWidth, int paddingHeight, int paddingWidth, int strideHeight, int strideWidth,
    int dilationHeight, int dilationWidth, int resultHeight, int resultWidth, int resObjCnt ) :
    mathEngine( _mathEngine ),
    ChCnt( channelCount ),
    FltH( filterHeight ),
    FltW( filterWidth ),
    SrcH( sourceHeight ),
    SrcW( sourceWidth ),
    PaddingH( paddingHeight ),
    PaddingW( paddingWidth ),
    StrideH( strideHeight ),
    StrideW( strideWidth ),
    DilationH( dilationHeight ),
    DilationW( dilationWidth ),
    ResH( resultHeight ),
    ResW( resultWidth ),
    ResObjCnt( resObjCnt ),
    jitIsInited( false ),
    src( nullptr ),
    flt( nullptr ),
    freeTerm( nullptr ),
    res( nullptr ),
    SrcLineStride( SrcW* ChCnt ),
    SrcXStep( StrideW* ChCnt ),
    SrcYStep( StrideH* SrcLineStride ),
    SrcXDilation( DilationW* ChCnt ),
    SrcYDilation( DilationH* SrcLineStride ),
    SrcXWindowSize( FltW* SrcXDilation ),
    ResLineStride( ResW* FltCnt ),
    NarrowBatchProcessSize( getNarrowBatchProcessSize() ),
    WideBatchProcessSize( getWideBatchProcessSize() )
{
    // // Initialize PixelOffsetResStepsX, PixelOffsetResStepsY, SrcPixelsOffset and FltPixelsOffset
    fillPixelOffset();
}

template<int FltCnt>
void CBlobConvolution<FltCnt>::ProcessConvolution(
    int threadCount, const float* sourceData, const float* filterData, const float* freeTermData, float* resultData )
{
    CFloatHandleStackVar filterTempBuffer( *mathEngine, FltW * FltH * FltCntM8 * ChCnt );
    CFloatHandleStackVar freeTermTempBuffer( *mathEngine, FltCntM8 );

    src = sourceData;
    // Filter offset also are calculated from center
    flt = rearrangeFilter( filterData, filterTempBuffer ) + ( FltW * FltH ) / 2 * ChCnt * FltCntM8;
    freeTerm = rearrangeFreeTerm( freeTermData, freeTermTempBuffer );
    res = resultData;

    if( !jitIsInited ) {
        initJitCodes();
        jitIsInited = true;
    }

    const int SrcObjSize = SrcW * SrcH * ChCnt;
    const int ResObjSize = ResW * ResH * FltCnt;
    const int ResRowCount = ResObjCnt * ResH;
    const int curThreadCount = IsOmpRelevant( ResRowCount, ResRowCount * ResW * FltCnt * FltW * FltH * ChCnt ) ? threadCount : 1;

    // Coordinates of the most top and left position of the center of the filter over the source image.
    const int srcXOffset = FltW / 2 * DilationW - PaddingW;
    const int srcYOffset = FltH / 2 * DilationH - PaddingH;

    NEOML_OMP_NUM_THREADS( curThreadCount )
    {
        // Index of row in whole result array
        int rowIdx;
        // Count of rows for current thread
        int rowCount;
        if( OmpGetTaskIndexAndCount( ResRowCount, rowIdx, rowCount ) ) {

            while( rowCount > 0 ) {
                // Index of result image in output batch
                int resIdx = rowIdx / ResH;
                // Offset in current result image
                int ryStart = rowIdx % ResH;
                // Number of rows for processing ( or number of rows till the end of current result image ).
                int ryCount = min( ResH - ryStart, rowCount );
                rowIdx += ryCount;
                rowCount -= ryCount;

                // Pointers to src and res for current thread
                const float* realSrcStart = src + resIdx * SrcObjSize + srcXOffset * ChCnt;
                float* realResStart = res + resIdx * ResObjSize;

                // Iterate through result, left->right, top->bottom
                const int currentRH = min( ResH, ryStart + ryCount );
                int ry = ryStart;
                int yStep = 0;

                // Iterate through all combination of intersections
                for( int yStepIndex = 0; yStepIndex < PixelOffsetResStepsWidthY.size(); yStepIndex++ ) {

                    // Last index of res for current intersection.
                    yStep += PixelOffsetResStepsWidthY[yStepIndex];
                    // Process up to current step or up to and of current butch
                    int ryEnd = min( yStep, currentRH );
                    for( ; ry < ryEnd; ) {
                        const float* srcPtr = realSrcStart + srcYOffset * SrcLineStride + ry * SrcYStep;
                        float* resPtr = realResStart + ry * ResLineStride;
                        bool useNarrowProcessing = ryEnd - ry >= NarrowBatchProcessSize.Height;

                        jitCodes[yStepIndex]->Run( useNarrowProcessing, srcPtr, flt, freeTerm, resPtr );
                        
                        ry += useNarrowProcessing ? NarrowBatchProcessSize.Height : WideBatchProcessSize.Height;
                    }
                }
            }
        }
    }
}

template<int FltCnt>
inline typename CBlobConvolution<FltCnt>::CSize CBlobConvolution<FltCnt>::getWideBatchProcessSize()
{
    return { WideBatchKernelHeight, WideBatchKernelWidth };
}

template<int FltCnt>
inline typename CBlobConvolution<FltCnt>::CSize CBlobConvolution<FltCnt>::getNarrowBatchProcessSize()
{
    // Disable narrow processing by default
    return { NarrowBatchKernelHeight, NarrowBatchKernelWidth };
}

template<int FltCnt>
inline void CBlobConvolution<FltCnt>::initJitCodes()
{
    jitCodes.resize( PixelOffsetResStepsWidthY.size() );
    for( int yStepIndex = 0; yStepIndex < PixelOffsetResStepsWidthY.size(); yStepIndex++ ) {
        jitCodes[yStepIndex] = std::unique_ptr<CJitConvolution>( new CJitConvolution( *this, yStepIndex ) );
    }
}

template<int FltCnt>
const float* CBlobConvolution<FltCnt>::rearrangeFilter( const float* filterData, CFloatHandleStackVar& filterTempBuffer )
{
    // Rearrange filter data.
    // Initial packing:
    // Filter[0] Pixel[0] Channel[0-23]
    // Filter[0] Pixel[1] Channel[0-23]
    // ...
    // Filter[0] Pixel[8] Channel[0-23]
    // Filter[1] Pixel[0] Channel[0-23]
    // ...
    // Filter[23] Pixel[8] Channel[0-23]
    //
    // 1. Result packing for case when FltCnt == FltCntM8 (for example: 24):
    // Pixel[0] Channel[0] Filter[0-23]
    // Pixel[0] Channel[1] Filter[0-23]
    // ...
    // Pixel[0] Channel[23] Filter[0-23]
    // Pixel[1] Channel[0] Filter[0-23]
    // ...
    // Pixel[8] Channel[23] Filter[0-23]
    //
    // 2. Result packing for case when FltCnt != FltCntM8 (for example: 18):
    // Pixel[0] Channel[0] Filter[0-17] Filter[0-5]
    // Pixel[0] Channel[1] Filter[0-23] Filter[0-5]
    // ...
    // Pixel[0] Channel[23] Filter[0-23] Filter[0-5]
    // Pixel[1] Channel[0] Filter[0-23] Filter[0-5]
    // ...
    // Pixel[8] Channel[23] Filter[0-23] Filter[0-5]

    float* resFilterStartPtr = static_cast< float* >( mathEngine->GetBuffer( filterTempBuffer.GetHandle(), 0, filterTempBuffer.Size() * sizeof( float ), false ) );
    float* resFilter = resFilterStartPtr;
    ASSERT_EXPR( reinterpret_cast< uintptr_t >( resFilter ) % AvxAlignment == 0 );
    for( int y = 0; y < FltH; y++ ) {
        for( int x = 0; x < FltW; x++ ) {
            for( int c = 0; c < ChCnt; c++ ) {
                const float* srcFilter = filterData + ( x + y * FltW ) * ChCnt + c;
                for( int f = 0; f < FltCnt; f++ ) {
                    resFilter[f] = *srcFilter;
                    srcFilter += FltW * FltH * ChCnt;
                }
                for( int f = FltCnt; f < FltCntM8; f++ ) {
                    resFilter[f] = resFilter[f % FltCnt];
                }
                resFilter += FltCntM8;
            }
        }
    }

    return resFilterStartPtr;
}

template<int FltCnt>
const float* CBlobConvolution<FltCnt>::rearrangeFreeTerm( const float* freeTermData, CFloatHandleStackVar& freeTermTempBuffer )
{
    if( freeTermData == nullptr ) {
        return nullptr;
    }

    float* resFreeTermStartPtr = static_cast< float* >( mathEngine->GetBuffer( freeTermTempBuffer.GetHandle(), 0, freeTermTempBuffer.Size() * sizeof( float ), false ) );
    float* resFreeTerm = resFreeTermStartPtr;
    ASSERT_EXPR( reinterpret_cast< uintptr_t >( resFreeTerm ) % AvxAlignment == 0 );

    for( int f = 0; f < FltCntM8; f++ ) {
        *resFreeTerm++ = freeTermData[f % FltCnt];
    }
    return resFreeTermStartPtr;
}

template<int FltCnt>
std::vector<int> CBlobConvolution<FltCnt>::getPixelOffsetSrcSteps( int srcDim, int fDim, int dDim, int sDim, int pDim )
{
    vector<int> ret( fDim );
    const int halfFDim = fDim / 2;

    // First offset of center of the filter window (Take in consideration paddings)
    const int firstOffset = halfFDim * dDim - pDim;
    const int lastSrcPixelIdx = srcDim - 1;
    // Last offset of center of the filter window (Take in consideration paddings)
    // (lastSrcPixelIdx - 2 * firstOffset) - width of window
    const int lastOffset = firstOffset + ( lastSrcPixelIdx - 2 * firstOffset ) / sDim * sDim;
    ret[0] = firstOffset;

    for( int i = 1; i <= halfFDim; i++ ) {
        // up to middle
        ret[i] = firstOffset + ( i * dDim - firstOffset + sDim - 1 ) / sDim * sDim;
    }


    for( int i = fDim - 1, j = 1; i > fDim / 2; i--, j++ ) {
        // from last to next to middle
        ret[i] = ( ( ( srcDim - j * dDim ) - firstOffset ) + sDim - 1 ) / sDim * sDim + firstOffset;
    }

    sort( ret.begin(), ret.end() );

    // Remove out of range and repeated items
    auto start = ret.begin();
    while( *start < 0 ) start++;
    auto end = start;
    auto tempIt = end + 1;
    int lastSrcDim = srcDim - firstOffset - 1;
    while( tempIt != ret.end() && *tempIt <= lastSrcDim ) {
        if( *tempIt != *end ) {
            int temp = *tempIt;
            *( ++end ) = temp;
        }
        tempIt++;
    }
    end++;

    return vector<int>( start, end );
}

template<int FltCnt>
void CBlobConvolution<FltCnt>::fillPixelOffset()
{
    using namespace std;
    vector<int> pixelOffsetSrcStepsX = getPixelOffsetSrcSteps( SrcW, FltW, DilationW, StrideW, PaddingW );
    vector<int> pixelOffsetSrcStepsY = getPixelOffsetSrcSteps( SrcH, FltH, DilationH, StrideH, PaddingH );

    // Calculate offset on the source image where intersection of filter and image is changed.
    auto getPixelOffsetResStepsWidth = []( const std::vector<int>& pixelOffsetSrcSteps, int srcDim, int fDim, int dDim, int sDim, int pDim )
    {
        vector<int> ret( pixelOffsetSrcSteps.size() );
        const int firstOffset = fDim / 2 * dDim - pDim;
        const int lastSrcPixelIdx = srcDim - 1;
        const int lastOffset = firstOffset + ( lastSrcPixelIdx - 2 * firstOffset ) / sDim * sDim;

        int i = 0;
        for( ; i < ret.size() - 1; i++ ) {
            ret[i] = ( pixelOffsetSrcSteps[i + 1] - pixelOffsetSrcSteps[i] ) / sDim;
        }
        ret[i] = ( lastOffset - pixelOffsetSrcSteps[i] ) / sDim + 1;

        return ret;
    };

    PixelOffsetResStepsWidthX = getPixelOffsetResStepsWidth( pixelOffsetSrcStepsX, SrcW, FltW, DilationW, StrideW, PaddingW );
    PixelOffsetResStepsWidthY = getPixelOffsetResStepsWidth( pixelOffsetSrcStepsY, SrcH, FltH, DilationH, StrideH, PaddingH );

    // Get size of intersection of filter window and source image
    auto getFilterWindowSize = []( const vector<int>& pixelOffsetSrcSteps, int srcDim, int fDim, int dDim ) -> vector<pair<int, int>> {
        // first - count of items in filter from center to top
        // second - count of items in filter from center to bottom
        vector<pair<int, int>> ret( pixelOffsetSrcSteps.size() );
        for( int i = 0; i < pixelOffsetSrcSteps.size(); i++ ) {
            const int halfFDim = fDim / 2;
            ret[i] = make_pair(
                min( pixelOffsetSrcSteps[i] / dDim, halfFDim ),
                min( ( ( srcDim - 1 ) - pixelOffsetSrcSteps[i] ) / dDim, halfFDim ) );
        }
        return ret;
    };

    vector<pair<int, int>> offsetSizeX = getFilterWindowSize( pixelOffsetSrcStepsX, SrcW, FltW, DilationW );
    vector<pair<int, int>> offsetSizeY = getFilterWindowSize( pixelOffsetSrcStepsY, SrcH, FltH, DilationH );

    // Calculate resulted offsets of pixels in window.
    auto fillPixelOffset = [&]( size_t hStride, size_t wStride ) ->vector<vector<int>> {
        vector<vector<int>> offsets( offsetSizeX.size() * offsetSizeY.size() );
        auto it = offsets.begin();

        for( const auto& y : offsetSizeY ) {
            for( const auto& x : offsetSizeX ) {
                it->resize( ( x.first + x.second + 1 ) * ( y.first + y.second + 1 ) );
                auto it_offt = it->begin();
                for( int i = -y.first; i <= y.second; i++ ) {
                    for( int j = -x.first; j <= x.second; j++ ) {
                        *it_offt++ = static_cast< int >( i * hStride + j * wStride );
                    }
                }
                it++;
            }
        }
        return offsets;
    };

    SrcPixelsOffset = fillPixelOffset( SrcYDilation, SrcXDilation );
    FltPixelsOffset = fillPixelOffset( FltW * ChCnt * FltCntM8, ChCnt * FltCntM8 );

}

} // namespace NeoML
