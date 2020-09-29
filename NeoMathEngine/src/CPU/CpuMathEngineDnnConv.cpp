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

#include <common.h>
#pragma hdrstop

#include <chrono>
#include <map>
#include <vector>
#include <sstream>
#include <iomanip>
#include <list>
#include <mutex>

using namespace std::chrono;
#include <CpuMathEngine.h>
#include <float.h>
#include <CpuMathEngineOmp.h>
#include <MathEngineCommon.h>
#include <MemoryHandleInternal.h>
#include <MathEngineDnnConv.h>
#include <CpuMathEnginePrivate.h>

class CTimer {
public:
	CTimer( bool start = false ) : CTimer( "", start ) {}

	CTimer( const char* _name, bool start = false ) : name( _name ), timeDelay( 0 ), count(0), isStarted( false ) {
		if( start ) {
			Start();
		}
	}
	~CTimer() {
		if( isStarted ) {
			Stop();
		}
		if( !name.empty() ) {
			const std::lock_guard<std::mutex> lock( TimersGuard );
			auto& timer = Timers[name];
			timer.delay += timeDelay;
			timer.count += count;
		}
	}
	void Clear() {
		 timeDelay = nanoseconds::zero();
		 count = 0;
		 isStarted = false;
	}

	void Start() {
		//assert( !isStarted );
		startTime = high_resolution_clock::now();
		isStarted = true;
	}
	void Stop() {
		//assert( isStarted );
		auto stopTime = high_resolution_clock::now();
		timeDelay += duration_cast<nanoseconds>( stopTime - startTime );
		count++;
		isStarted = false;
	}

	float GetTimeInMs() const {
		auto currentTime = isStarted ? high_resolution_clock::now() - startTime : timeDelay;
		return currentTime.count() / 1e6;
	}

	static string PrintTimers() {
		using namespace std;
		const std::lock_guard<std::mutex> lock( TimersGuard );
		stringstream ss;
		ss << endl << "_tmrs_;Timer name;Total, ms;Avrg, ms;Count" << endl;

		for( auto& timer : Timers ) {
			auto& timerStruct = timer.second;
			ss << "_tmrs_;"
			   << timer.first << ";"
			   << timerStruct.delay.count() / 1000000 << ";"
			   << setprecision(3) << timerStruct.delay.count() / 1e6 / timerStruct.count << ";" << fixed
			   << timerStruct.count << endl;
		}
		Timers.clear();
		return ss.str();
	}

private:
	struct CTimerStruct {
		CTimerStruct() : delay( 0 ), count ( 0 ) {}

		nanoseconds delay;
		int64_t count;
	};

	std::string name;
	system_clock::time_point startTime;
	nanoseconds timeDelay;
	int64_t count;
	bool isStarted;
	static std::map<std::string, CTimerStruct> Timers;
	static std::mutex TimersGuard;
};

std::map<std::string, CTimer::CTimerStruct> CTimer::Timers;
std::mutex CTimer::TimersGuard;

NEOMATHENGINE_API std::string PrintTimers() {
	return CTimer::PrintTimers();
}

class CAlgoInfo {
public:

	struct CInfo {
		std::vector<CTimer> Timers;
		std::vector<int> Dimentions;
	};


	static void AddFastAlgo( CInfo&& fastAlgoInfo ) {
		const std::lock_guard<std::mutex> lock( AlgoInfoGuard );
		FastAlgoInfo.push_back( fastAlgoInfo );
	}

	static void AddAlgo0Info( CInfo&& algo0Info ) {
		return;
		const std::lock_guard<std::mutex> lock( AlgoInfoGuard );
		Algo0Info.push_back( algo0Info );
	}

	static void AddAlgo0VsFastAlgoInfo( CInfo&& algo0VsFastAlgoInfo ) {
		const std::lock_guard<std::mutex> lock( AlgoInfoGuard );
		Algo0VsFastAlgoInfo.push_back( algo0VsFastAlgoInfo );
	}

	static std::string PrintAlgoInfo() {
		const std::lock_guard<std::mutex> lock( AlgoInfoGuard );

		using namespace std;
		stringstream ss;

		printInfo( ss, Algo0VsFastAlgoInfo, "Algo0;FastAlgo;SW;D;S", "_info_a0vsfa" );
		printInfo( ss, FastAlgoInfo, "full;t1;t2;t3", "_info_fastAlgo" );
		printInfo( ss, Algo0Info, "full;t0;t1;t2", "_info_algo0" );

		return ss.str();

	}
private:
	static void printInfo( stringstream& ss, const std::list<CInfo>& info, const char* head, const char* tag ) {
			ss << endl <<  tag << head << endl;
			for( auto& i : info ) {
				ss << tag << ";";
				for( auto& t: i.Timers ) {
					ss << t.GetTimeInMs() << ";";
				}
				for( auto& d: i.Dimentions ) {
					ss << d << ";";
				}
				ss << endl;
			}
	}
	static std::list<CInfo> Algo0VsFastAlgoInfo;
	static std::list<CInfo> FastAlgoInfo;
	static std::list<CInfo> Algo0Info;
	static std::mutex AlgoInfoGuard;
};
std::list<CAlgoInfo::CInfo> CAlgoInfo::Algo0VsFastAlgoInfo;
std::list<CAlgoInfo::CInfo> CAlgoInfo::FastAlgoInfo;
std::list<CAlgoInfo::CInfo> CAlgoInfo::Algo0Info;
std::mutex CAlgoInfo::AlgoInfoGuard;

NEOMATHENGINE_API std::string PrintAlgoInfo() {
	return CAlgoInfo::PrintAlgoInfo();
}


namespace NeoML {

// The algorithm used to calculate a 2D convolution
enum TConvAlgo {
	CA_Auto,	// choose automatically
	CA_1,		// use a temporary matrix to store the data in another order
	CA_2,		// work with the data directly (only for stride = 1 and padding = 0)
				// most efficient when the image is large and especially when it has many channels

	CA_1x1		// for convolution with a 1*1 filter, no padding and dilation (both 2D and 3D)
};

const int BlobConvolutionCacheSize = 256 * 1024;

// Convolution descriptor
struct CCpuConvolutionDesc : public CCommonConvolutionDesc {
	TConvAlgo ForwardAlgo;
	TConvAlgo BackwardAlgo;

	CCpuConvolutionDesc( const CBlobDesc& source, const CBlobDesc& result, const CBlobDesc& filter,
			int paddingHeight, int paddingWidth, int strideHeight, int strideWidth, int dilationHeight, int dilationWidth ) :
		CCommonConvolutionDesc( source, filter, result, paddingHeight, paddingWidth, strideHeight, strideWidth, dilationHeight, dilationWidth ),
		ForwardAlgo( getActualForwardAlgo() ),
		BackwardAlgo( getActualBackwardAlgo() )
	{
	}

	TConvAlgo getActualForwardAlgo() const;
	TConvAlgo getActualBackwardAlgo() const;
};

// Gets the algorithm to be used for this convolution
inline TConvAlgo CCpuConvolutionDesc::getActualForwardAlgo() const
{
	if( PaddingHeight == 0 && PaddingWidth == 0
		&& DilationHeight == 1 && DilationWidth == 1
		&& Filter.ObjectSize() == Filter.Channels() )
	{
		return CA_1x1;
	}

	if( DilationHeight == 1 && DilationWidth == 1 && StrideHeight == 1 && StrideWidth == 1 ) {
		if( PaddingHeight > 0 || PaddingWidth > 0 ) {
			if( ( Source.Height() >= 64 && Source.Width() >= 64 && Source.Depth() * Source.Channels() >= 8 ) ||
				( Source.Height() >= 32 && Source.Width() >= 32 && Source.Depth() * Source.Channels() >= 16 ) )
			{
				return CA_2;
			}
		} else {
			if( ( Source.Height() >= 64 && Source.Width() >= 64 && Source.Depth() * Source.Channels() >= 4 ) ||
				( Source.Height() >= 32 && Source.Width() >= 32 && Source.Depth() * Source.Channels() >= 8 ) )
			{
				return CA_2;
			}
		}
	}
	return CA_1;
}

inline TConvAlgo CCpuConvolutionDesc::getActualBackwardAlgo() const
{
	TConvAlgo ret = getActualForwardAlgo();
	if( ret == CA_2 && ( PaddingHeight != 0 || PaddingWidth != 0 ) ) {
		ret = CA_1;
	}

	return ret;
}

// Returns the descriptor of the "flattened" blob with depth == 1 and channels = desc.depth * desc.channels
static inline CBlobDesc flatten( const CBlobDesc& desc )
{
	CBlobDesc res = desc;
	res.SetDimSize( BD_Depth, 1 );
	res.SetDimSize( BD_Channels, desc.Channels() * desc.Depth() );
	return res;
}

///////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////
CConvolutionDesc* CCpuMathEngine::InitBlobConvolution( const CBlobDesc& source, int paddingHeight, int paddingWidth,
	int strideHeight, int strideWidth, int dilationHeight, int dilationWidth, const CBlobDesc& filter, const CBlobDesc& result )
{
	ASSERT_EXPR( strideHeight > 0 );
	ASSERT_EXPR( strideWidth > 0 );
	ASSERT_EXPR( paddingHeight >= 0 );
	ASSERT_EXPR( paddingWidth >= 0 );
	ASSERT_EXPR( dilationHeight > 0 );
	ASSERT_EXPR( dilationWidth > 0 );
	ASSERT_EXPR( source.Channels() == filter.Channels() );
	ASSERT_EXPR( source.Depth() == filter.Depth() );
	ASSERT_EXPR( filter.Height() <= source.Height() + 2 * paddingHeight );
	ASSERT_EXPR( filter.Width() <= source.Width() + 2 * paddingWidth );
	ASSERT_EXPR( filter.BatchLength() == 1 );
	ASSERT_EXPR( result.BatchLength() == source.BatchLength() );
	ASSERT_EXPR( result.BatchWidth() == source.BatchWidth() );
	ASSERT_EXPR( result.Height() == 1 + ( source.Height() -
		( filter.Height() - 1 ) * dilationHeight + 2 * paddingHeight - 1 ) / strideHeight );
	ASSERT_EXPR( result.Width() == 1 + ( source.Width() -
		( filter.Width() - 1 ) * dilationWidth + 2 * paddingWidth - 1 ) / strideWidth );
	ASSERT_EXPR( result.Channels() == filter.BatchWidth() );
	ASSERT_EXPR( result.Depth() == 1 );

	CCpuConvolutionDesc* desc = new CCpuConvolutionDesc( source, filter, result,
		paddingHeight, paddingWidth, strideHeight, strideWidth, dilationHeight, dilationWidth );
	return desc;
}

// Creates a temporary blob with reordered input data that will be used to calculate convolution
// This method allows for nonzero dilation
void CCpuMathEngine::createDilationTemporaryBlob( const CCpuConvolutionDesc& desc, const CFloatHandle& inputData, int inputBatch,
	int outputColumnStart, int outputColumnCount, const CFloatHandle& temporaryBlob )
{
	const CBlobDesc& inputBlob = desc.Source;
	const CBlobDesc& filter = desc.Filter;
	const CBlobDesc& output = desc.Result;

	CConstFloatHandle inputBlobPtr = inputData + inputBlob.ObjectSize() * inputBatch;
	CFloatHandle tempBlobPtr = temporaryBlob;

	const int vectorSize = inputBlob.Depth() * inputBlob.Channels();
	if( desc.PaddingHeight > 0 || desc.PaddingWidth > 0 ) {
		// Padding is emulated by first filling the tempBlob by the padding value
		// and then writing over the required positions with the input data
		vectorFill( tempBlobPtr, 0.0f, filter.Height() * filter.Width() * vectorSize * output.Height() * outputColumnCount );
	}

	for( int outputColumn = outputColumnStart; outputColumn < outputColumnStart + outputColumnCount; outputColumn++ ) {
		const int leftPos = -desc.PaddingWidth + outputColumn * desc.StrideWidth;
		if( leftPos + ( filter.Width() - 1 ) * desc.DilationWidth < 0 || leftPos >= inputBlob.Width() ) {
			// The current column is all padding
			continue;
		}

		for( int outputRow = 0; outputRow < output.Height(); outputRow++ ) {
			const int topPos = -desc.PaddingHeight + outputRow * desc.StrideHeight;
			if( topPos + ( filter.Height() - 1 ) * desc.DilationHeight < 0 || topPos >= inputBlob.Height() ) {
				// The current row is all padding
				continue;
			}

			CFloatHandle tempRow = tempBlobPtr + ( ( outputColumn - outputColumnStart ) * output.Height() + outputRow )
				* filter.Height() * filter.Width() * vectorSize;

			for( int filterRow = 0; filterRow < filter.Height(); filterRow++ ) {
				const int verticalPos = topPos + desc.DilationHeight * filterRow;
				if( verticalPos < 0 || verticalPos >= inputBlob.Height() ) {
					// The current filter row only intersects with padding
					continue;
				}

				for( int filterColumn = 0; filterColumn < filter.Width(); filterColumn++ ) {
					const int horizontalPos = leftPos + desc.DilationWidth * filterColumn;
					if( horizontalPos < 0 || horizontalPos >= inputBlob.Width() ) {
						// The current element is padding
						continue;
					}

					const float* inputVector = GetRaw( inputBlobPtr ) + ( inputBlob.Width() * verticalPos + horizontalPos ) * vectorSize;
					float* tempVector = GetRaw( tempRow ) + ( filterRow * filter.Width() + filterColumn ) * vectorSize;
					dataCopy( tempVector, inputVector, vectorSize );
				}
			}
		}
	}
}

template<class TConvolutionDesc>
void CCpuMathEngine::createTemporaryBlob( const TConvolutionDesc& desc, const CFloatHandle& inputData,
	int inputBatch, int outputRowStart, int outputRowCount, const CFloatHandle& tempBlob )
{
	const CBlobDesc& inputBlob = desc.Source;
	const CBlobDesc& filter = desc.Filter;
	const CBlobDesc& output = desc.Result;

	const int inputChannelsCount = inputBlob.Depth() * inputBlob.Channels();
	const int windowRowSize = filter.Width() * inputChannelsCount;
	const int inputRowSize = inputBlob.Width() * inputBlob.Depth() * inputBlob.Channels();
	const int outputRowEnd = outputRowStart + outputRowCount;

	const float* inputBlobPtr = GetRaw( inputData ) + inputBlob.ObjectSize() * inputBatch;
	float* tempBlobPtr = GetRaw( tempBlob );

	if( desc.PaddingHeight > 0 || desc.PaddingWidth > 0 ) {
		// Padding is emulated by first filling the tempBlob by the padding value
		// and then writing over the required positions with the input data
		NeoML::vectorFill( tempBlobPtr, 0.0f, filter.Height() * filter.Width() * inputBlob.Depth() * inputBlob.Channels() *
			output.Height() * outputRowCount );
	}

	// The input blob height - the number of filter windows that fit horizontally
	for(int j = outputRowStart; j < outputRowEnd; j++) {
		// Skip the top of the first window (padding)
		tempBlobPtr += desc.PaddingHeight * windowRowSize;
		// Calculate padding on the left and right
		int paddingLeft = (desc.PaddingWidth - j * desc.StrideWidth) * inputChannelsCount;
		if(paddingLeft < 0) {
			paddingLeft = 0;
		}
		int paddingRight = (filter.Width() - desc.PaddingWidth + j * desc.StrideWidth - inputBlob.Width()) *
			inputChannelsCount;
		if(paddingRight < 0) {
			paddingRight = 0;
		}
		// Copy the vertical strip of windows in a cycle
		// The start of the top window of the strip
		const float* currentWindowStart = inputBlobPtr + (j * desc.StrideWidth - desc.PaddingWidth) * inputChannelsCount;
		// Copy the first window
		for(int k = 0; k < filter.Height() - desc.PaddingHeight; k++) {
			if( k < inputBlob.Height() ) {
				dataCopy( tempBlobPtr + paddingLeft, currentWindowStart + paddingLeft,
					( windowRowSize - paddingLeft - paddingRight ) );
				currentWindowStart += inputRowSize;
			}
			tempBlobPtr += windowRowSize;
		}
		// The input blob width - the number of filter windows that fit vertically
		for(int k = 1; k < output.Height(); k++) {
			if(filter.Height() >= desc.StrideHeight) {
				// The size of intersection for two vertically adjacent windows
				const int windowsIntersection = windowRowSize * (filter.Height() - desc.StrideHeight);

				// If stride is smaller than the filter size, copy the adjacent filters intersection
				dataCopy(tempBlobPtr, tempBlobPtr - windowsIntersection, windowsIntersection);
				tempBlobPtr += windowsIntersection;
			} else {
				// If stride is larger than filter size, skip the rows that will not be in the next filter
				currentWindowStart += (desc.StrideHeight - filter.Height()) * inputRowSize;
			}

			// The lower filter boundary - ( top padding + the input image height )
			// If this number is greater than 0, the filter intersects with the bottom padding
			int paddingBottom = filter.Height() + k * desc.StrideHeight - desc.PaddingHeight - inputBlob.Height();
			if(paddingBottom < 0) {
				// If this number is smaller than 0, the filter does not intersect with the bottom padding
				paddingBottom = 0;
			}
			if(paddingBottom > min(desc.StrideHeight, filter.Height())) {
				// The whole area to be copied next belongs to the bottom padding
				paddingBottom = min(desc.StrideHeight, filter.Height());
			}
			// The paddingBottom now has only the bottom padding rows that are in the filter area

			// Copy the rows that are in filter area
			// If strideHeight <= filterHeight the intersection has already been copied above
			// and we only need to copy additional strideHeight lower rows
			// If strideHeight > filterHeight the rows to be ignored have been skipped already
			// and we need to copy filterHeight lower rows
			// The intersection with the bottom padding does not need copying
			// because we've already filled temporaryBlob with the padding value
			for(int l = 0; l < min(desc.StrideHeight, filter.Height()) - paddingBottom; l++) {
				dataCopy(tempBlobPtr + paddingLeft, currentWindowStart + paddingLeft,
					(windowRowSize - paddingLeft - paddingRight));
				currentWindowStart += inputRowSize;
				tempBlobPtr += windowRowSize;
			}

			// temporaryBlob already filled with the padding value, so we only need to
			// offset the pointer by the number of bottom padding elements that fit into the filter area
			tempBlobPtr += paddingBottom * windowRowSize;
		}
	}
}

void CCpuMathEngine::transposeResult( const CCpuConvolutionDesc& desc, const CConstFloatHandle& outputTransposedData,
	int batch, int resultStart, int resultCount, const CFloatHandle& resultData )
{
	const CBlobDesc& result = desc.Result;

	int resultPixelSize = result.Depth() * result.Channels();
	int resultRowSize = resultPixelSize * result.Width();
	const float* inPtr = GetRaw( outputTransposedData );
	float* resultPtr = GetRaw( resultData ) + batch * result.ObjectSize() + resultStart * resultPixelSize;
	for(int i = 0; i < resultCount; ++i) {
		float* outRowPtr = resultPtr;
		for(int j = 0; j < result.Height(); ++j) {
			dataCopy(outRowPtr, inPtr, resultPixelSize);
			outRowPtr += resultRowSize;
			inPtr += resultPixelSize;
		}
		resultPtr += resultPixelSize;
	}
}

static inline void calcPaddings( const CCpuConvolutionDesc& desc, int width, int& startPaddingSize, int& endPaddingSize )
{
	int startPos = -desc.PaddingWidth + width * desc.StrideWidth;
	startPaddingSize = min( desc.Filter.Width(), ( startPos < 0 ) ? 1 + ( -startPos - 1 ) / desc.DilationWidth : 0 );

	int endPos = -desc.PaddingWidth + width * desc.StrideWidth + desc.DilationWidth * ( desc.Filter.Width() - 1 );
	endPaddingSize = ( desc.Source.Width() > endPos ) ? 0 :
		min( ( endPos - desc.Source.Width() ) / desc.DilationWidth + 1, desc.Filter.Width() );
}

void CCpuMathEngine::fillTempData( const CFloatHandle& sourceData, const CFloatHandle& tempData, const CCpuConvolutionDesc& desc, int start, int count )
{
	const int channelsCount = desc.Filter.Depth() * desc.Filter.Channels();
	const int filterLineSize = desc.Filter.Width() * channelsCount;
	const int resultG = desc.Result.Width() * desc.Result.Height();

	for( int index = start; index < count + start; index++ ) {
		const int batch = index / resultG;
		const int height = ( index - batch * resultG ) / desc.Result.Width();
		const int width = ( index - batch * resultG ) % desc.Result.Width();

		int startPaddingSize = 0;
		int endPaddingSize = 0;
		calcPaddings( desc, width, startPaddingSize, endPaddingSize );
		const int dataSize = desc.Filter.Width() - startPaddingSize - endPaddingSize;

		const int sourceHeight = -desc.PaddingHeight + height * desc.StrideHeight;
		const int sourceWidth = -desc.PaddingWidth + width * desc.StrideWidth + startPaddingSize * desc.DilationWidth;

		const float* sourceDataPtr = GetRaw(sourceData) + batch * desc.Source.ObjectSize() + ( sourceHeight * desc.Source.Width() + sourceWidth ) * channelsCount;
		float* tempStartPaddingPtr = GetRaw(tempData) + ( index - start ) * desc.Filter.ObjectSize();
		float* tempDataPtr = tempStartPaddingPtr + startPaddingSize * channelsCount;
		float* tempEndPaddingPtr = tempDataPtr + dataSize * channelsCount;

		for( int h = 0; h < desc.Filter.Height(); h++ ) {
			if( 0 <= sourceHeight + h * desc.DilationHeight && sourceHeight + h * desc.DilationHeight < desc.Source.Height() ) {
				if( startPaddingSize > 0 ) {
					NeoML::vectorFill( tempStartPaddingPtr, 0.0, startPaddingSize * channelsCount );
				}

				if( desc.DilationWidth == 1 ) {
					if( dataSize > 0 ) {
						dataCopy( tempDataPtr, sourceDataPtr, dataSize * channelsCount );
					}
				} else {
					for( int i = 0; i < dataSize; i++ ) {
						dataCopy( tempDataPtr + i * channelsCount, sourceDataPtr + i * desc.DilationWidth * channelsCount, channelsCount );
					}
				}

				if( endPaddingSize > 0 ) {
					NeoML::vectorFill( tempEndPaddingPtr, 0.0, endPaddingSize * channelsCount );
				}
			} else {
				NeoML::vectorFill( tempStartPaddingPtr, 0.0, filterLineSize );
			}

			tempStartPaddingPtr += filterLineSize;
			tempDataPtr += filterLineSize;
			tempEndPaddingPtr += filterLineSize;
			sourceDataPtr += desc.DilationHeight * desc.Source.Width() * channelsCount;
		}
	}
}

inline int ceilTo( int val, int discret )
{
	if( val > 0 ) {
		return ( ( val + discret - 1 ) / discret ) * discret;
	}
	return ( val / discret ) * discret;
}

void CCpuMathEngine::blobConvolutionForwardAlgo0( const CCpuConvolutionDesc& desc, const CFloatHandle& sourceData,
	const CFloatHandle& filterData, const CFloatHandle* freeTermData, const CFloatHandle& resultData )
{
	CTimer t0;
	CTimer t1;
	CTimer t2;
	CTimer full;
	full.Start();
	const int resultItemCount = desc.Result.ObjectCount() * desc.Result.Width() * desc.Result.Height();
	const int curThreadCount = IsOmpRelevant( resultItemCount, static_cast<int64_t>( desc.Result.BlobSize() ) * desc.Filter.ObjectSize() ) ? threadCount : 1;
	const int cacheItemCount = max( 1, min( ceilTo( BlobConvolutionCacheSize / desc.Filter.ObjectSize(), 16 ), resultItemCount / curThreadCount ) );
	const int tempDataSize = curThreadCount * cacheItemCount * desc.Filter.ObjectSize();

	CFloatHandleStackVar tempData( mathEngine(), tempDataSize );

	NEOML_OMP_NUM_THREADS( curThreadCount )
	{
		const int filterObjectCount = desc.Filter.ObjectCount();
		const int filterObjectSize = desc.Filter.ObjectSize();
		CFloatHandle tempDataPtr = tempData + OmpGetThreadNum() * cacheItemCount * filterObjectSize;

		int start;
		int count;
		if( OmpGetTaskIndexAndCount( resultItemCount, start, count ) ) {
			int index = 0;
			while( index < count ) {
				const int size = min( count - index, cacheItemCount );

				t0.Start();
				fillTempData( sourceData, tempDataPtr, desc, start + index, size );
				t0.Stop();

				CFloatHandle resultDataPtr = resultData + ( start + index ) * filterObjectCount;

				t1.Start();
				multiplyMatrixByTransposedMatrix( tempDataPtr, size, filterObjectSize,
					filterObjectSize, filterData, filterObjectCount, filterObjectSize, resultDataPtr,
					filterObjectCount, resultItemCount * filterObjectCount );

				t1.Stop();
				if( freeTermData != 0 ) {
					t2.Start();
					addVectorToMatrixRows( resultDataPtr, resultDataPtr, size, filterObjectCount, filterObjectCount, filterObjectCount, *freeTermData );
					t2.Stop();
				}

				index += size;
			}
		}
	}
	full.Stop();
	CAlgoInfo::AddAlgo0Info( { { full, t0, t1, t2 },
		{ desc.Source.Height(), desc.Source.Width(), desc.Filter.Height(), desc.Filter.Width(), desc.DilationHeight, desc.StrideHeight, desc.Filter.Channels(), desc.Filter.ObjectCount() } } );
}

void CCpuMathEngine::blobConvolutionForwardAlgo1( const CCpuConvolutionDesc& desc, const CFloatHandle& sourceData,
	const CFloatHandle& filterData, const CFloatHandle* freeTermData, const CFloatHandle& resultData )
{
	const CBlobDesc& src = desc.Source;
	const CBlobDesc& fil = desc.Filter;
	const CBlobDesc& res = desc.Result;

	const int outputChannels = res.Depth() * res.Channels();
	const int outputTransposedDataRowSize = res.Height() * outputChannels;
	const int outputTransposedDataObjectSize = res.Width() * outputTransposedDataRowSize;
	const int tempBlobDataRowSize = res.Height() * fil.Height() * fil.Width() * src.Depth() * src.Channels();
	const int tempBlobDataObjectSize = res.Width() * tempBlobDataRowSize;

	const int curThreadCount = IsOmpRelevant( src.ObjectCount() * res.Width(),
		static_cast<int64_t>( src.BlobSize() ) * fil.BlobSize() ) ? threadCount : 1;
	const int tempObjectCount = min( src.ObjectCount(), curThreadCount );

	const int outputTransposedDataSize = tempObjectCount * outputTransposedDataObjectSize;
	const int tempBlobDataSize = tempObjectCount * tempBlobDataObjectSize;

	CFloatHandleStackVar stackBuffer( mathEngine(), outputTransposedDataSize + tempBlobDataSize );
	CFloatHandle outputTransposedData = stackBuffer;
	CFloatHandle tempBlobData = stackBuffer + outputTransposedDataSize;

	NEOML_OMP_NUM_THREADS( curThreadCount )
	{
		const CBlobDesc& source = desc.Source;
		const CBlobDesc& filter = desc.Filter;
		const CBlobDesc& result = desc.Result;

		int batchStart;
		int batchCount;
		int resultStart;
		int resultCount;
		if( OmpGetTaskIndexAndCount2D( source.ObjectCount(), result.Width(), batchStart, batchCount, resultStart, resultCount ) ) {
			for( int batch = batchStart; batch < batchStart + batchCount; batch++ ) {
				const int tempObjectIndex = source.ObjectCount() <= tempObjectCount ? batch : OmpGetThreadNum();
				CFloatHandle outputTransposedPtr = outputTransposedData + tempObjectIndex * outputTransposedDataObjectSize
					+ resultStart * outputTransposedDataRowSize;
				CFloatHandle tempBlobPtr = tempBlobData + tempObjectIndex * tempBlobDataObjectSize
					+ resultStart * tempBlobDataRowSize;

				// Fill the temporary matrix
				if( desc.DilationHeight > 1 || desc.DilationWidth > 1 ) {
					createDilationTemporaryBlob( desc, sourceData, batch, resultStart, resultCount, tempBlobPtr );
				} else {
					createTemporaryBlob( desc, sourceData, batch, resultStart, resultCount, tempBlobPtr );
				}

				// Apply the filter to the temporary matrix
				if( freeTermData != 0 ) {
					setVectorToMatrixRows( outputTransposedPtr, result.Height() * resultCount, outputChannels, *freeTermData );

					multiplyMatrixByTransposedMatrixAndAdd( GetRaw(tempBlobPtr), result.Height() * resultCount, filter.ObjectSize(),
						filter.ObjectSize(), GetRaw(filterData), filter.BatchWidth(), filter.ObjectSize(), GetRaw(outputTransposedPtr),
						filter.BatchWidth() );
				} else {
				multiplyMatrixByTransposedMatrix( tempBlobPtr, result.Height() * resultCount, filter.ObjectSize(),
					filter.ObjectSize(), filterData, filter.BatchWidth(), filter.ObjectSize(), outputTransposedPtr,
					filter.BatchWidth(), resultCount * outputTransposedDataRowSize );
				}

				// Transpose the result
				transposeResult( desc, outputTransposedPtr, batch, resultStart, resultCount, resultData );
			}
		}
	}
}

void CCpuMathEngine::BlobConvolution( const CConvolutionDesc& convDesc, const CFloatHandle& source,
	const CFloatHandle& filter, const CFloatHandle* freeTerm, const CFloatHandle& result )
{
	const CCpuConvolutionDesc& desc = static_cast<const CCpuConvolutionDesc&>( convDesc );

	switch( desc.ForwardAlgo ) {
		case CA_1:
		case CA_2:
		{
			const int algo0ThreadCount = IsOmpRelevant( desc.Result.ObjectCount() * desc.Result.Width() * desc.Result.Height(),
				static_cast<int64_t>( desc.Result.BlobSize() ) * desc.Filter.ObjectSize() ) ? threadCount : 1;

			const int algo1ThreadCount = IsOmpRelevant( desc.Result.ObjectCount() * desc.Result.Width(),
				static_cast<int64_t>( desc.Result.BlobSize() ) * desc.Filter.ObjectSize() ) ? threadCount : 1;
			const int64_t algo1DataSize = static_cast<int64_t>( desc.Result.Width() ) * desc.Result.Height() * desc.Filter.ObjectSize() + desc.Result.ObjectSize();

			CTimer t0;
			CTimer t1;

			if( ( desc.Filter.Channels() == 24 ) &&
				( desc.Filter.ObjectCount() == 24 ) &&
				( desc.Filter.Width() == 3 ) &&
				( desc.Filter.Height() == 3 ) &&
				CAvxDll::GetInstance().IsAvailable() ) {
				CFloatHandleStackVar R( mathEngine(), desc.Result.Width() * desc.Result.Height() * desc.Filter.ObjectCount() );

				t1.Start();
				CAvxDll::GetInstance().CallBlobConvolution_avx_f9x9_c24_fc24( mathEngine(), threadCount, desc, source, filter, freeTerm, R.GetHandle() );
				t1.Stop();
				t0.Start();
				if( min( desc.Result.ObjectCount(), algo1ThreadCount ) * algo1DataSize <= algo0ThreadCount * BlobConvolutionCacheSize ) {
					blobConvolutionForwardAlgo1( desc, source, filter, freeTerm, result );
				} else {
					blobConvolutionForwardAlgo0( desc, source, filter, freeTerm, result );
				}
				t0.Stop();
				CAlgoInfo::AddAlgo0VsFastAlgoInfo( { { t0, t1 }, { desc.Source.Width(), desc.DilationWidth, desc.StrideWidth } } );
				float* f1 = GetRaw( R.GetHandle() );
				float* f2 = GetRaw( result);
				for( int i = 0; i < desc.Result.Width() * desc.Result.Height() * desc.Filter.ObjectCount(); i++ ) {
					const float e = 1e-4;
					const float sub = *f1++ - *f2++;
					ASSERT_EXPR( sub > -e && sub < e );
				}
			} else {
				if( min( desc.Result.ObjectCount(), algo1ThreadCount ) * algo1DataSize <= algo0ThreadCount * BlobConvolutionCacheSize ) {
					CTimer timer( "Algo1", true );
					blobConvolutionForwardAlgo1( desc, source, filter, freeTerm, result );
				} else {
					CTimer timer( "Algo0", true );
					blobConvolutionForwardAlgo0( desc, source, filter, freeTerm, result );
				}
			}
			break;
		}
		case CA_1x1:
			{
				bool needsFlatten = desc.Source.Depth() != 1;

				blob3dConvolution1x1x1( needsFlatten ? flatten( desc.Source ) : desc.Source, needsFlatten ? flatten( desc.Filter ) : desc.Filter,
					desc.Result, desc.StrideHeight, desc.StrideWidth, 1, GetRaw( source ), GetRaw( filter ),
					freeTerm != 0 ? GetRaw( *freeTerm ) : 0, GetRaw( result ) );
				break;
			}
		default:
			ASSERT_EXPR( false );
	}
}

void CCpuMathEngine::backwardConvolutionAddFilterToOutput( const CCpuConvolutionDesc& desc, const CFloatHandle& temp,
	const CFloatHandle* freeTermData, const CFloatHandle& outputData )
{
	const CBlobDesc& input = desc.Result;
	const CBlobDesc& output = desc.Source;
	const CBlobDesc& filter = desc.Filter;

	int filterChannels = filter.Depth() * filter.Channels();

	int outputLineStart;
	int outputLineEnd;
	OmpGetTaskIndexAndCount( output.ObjectCount() * output.Height(), outputLineStart, outputLineEnd );
	outputLineEnd += outputLineStart;
	for( int step = outputLineStart; step < outputLineEnd; ++step ) {
		CFloatHandle outputDataPtr = outputData + step * output.Width() * output.Depth() * output.Channels();

		if( freeTermData != 0 ) {
			// Set the free term
			setVectorToMatrixRows( outputDataPtr, output.Width(), output.Depth() * output.Channels(), *freeTermData );
		} else {
			vectorFill( outputDataPtr, 0, output.Width() * output.Depth() * output.Channels() );
		}

		int batch = step / output.Height();
		int row = step % output.Height();
		int inputRowStart = (row + desc.PaddingHeight - filter.Height() + desc.StrideHeight) / desc.StrideHeight;
		if(inputRowStart < 0) {
			inputRowStart = 0;
		}
		int filterRowBackStart = row - inputRowStart * desc.StrideHeight + desc.PaddingHeight;
		if(0 > filterRowBackStart || filterRowBackStart >= filter.Height()) {
			continue;
		}
		int filterRowBackEnd = filter.Height() + row - output.Height() - desc.PaddingHeight;
		if(filterRowBackEnd < 0) {
			filterRowBackEnd = 0;
		}

		int inputRow = inputRowStart;
		for(int filterRow = filterRowBackStart;
			filterRow >= filterRowBackEnd;
			filterRow -= desc.StrideHeight, ++inputRow) {
			// The temp blob stores the filter rows multiplied by input; add them to the output rows in correct positions
			CConstFloatHandle tempRowData = temp + ( ( batch * input.Height() + inputRow )
				* input.Width() * filter.Height() + filterRow ) * filter.Width() * filterChannels;

			for(int col = -desc.PaddingWidth;
				col <= output.Width() + desc.PaddingWidth - filter.Width();
				col += desc.StrideWidth) {

				int tempRowDataShift = 0;
				int toCopy = filter.Width();
				int pos = col;
				if(pos < 0) {
					tempRowDataShift = -pos;
					toCopy += pos;
					pos = 0;
				}
				if(pos + toCopy > output.Width()) {
					toCopy = output.Width() - pos;
				}
				if( toCopy > 0 ) {
					toCopy *= filterChannels;
					tempRowDataShift *= filterChannels;
					float* outputVec = GetRaw(outputDataPtr) + pos * filterChannels;
					vectorAdd(outputVec, GetRaw(tempRowData) + tempRowDataShift, outputVec, toCopy);
				}
				tempRowData += filter.Height() * filter.Width() * filterChannels;
			}
		}
	}
}

void CCpuMathEngine::backwardDilationConvolutionAddFilterToOutput( const CCpuConvolutionDesc& desc, const CFloatHandle& temp,
	const CFloatHandle* freeTermData, const CFloatHandle& outputData )
{
	const CBlobDesc& source = desc.Result;
	const CBlobDesc& filter = desc.Filter;
	const CBlobDesc& output = desc.Source;

	ASSERT_EXPR( desc.DilationHeight >= 1 );
	ASSERT_EXPR( desc.DilationWidth >= 1 );
	ASSERT_EXPR( desc.DilationHeight + desc.DilationWidth > 2 );

	const int vectorSize = output.Depth() * output.Channels();
	int startRow;
	int rowCount;
	OmpGetTaskIndexAndCount( output.ObjectCount() * output.Height(), startRow, rowCount );

	const int totalFilterHeight = ( filter.Height() - 1 ) * desc.DilationHeight + 1;
	const int totalFilterWidth = ( filter.Width() - 1 ) * desc.DilationWidth + 1;

	for( int row = startRow; row < startRow + rowCount; row++ ) {
		// Separate calculations for each row

		// Find all filters that affect the row
		const int batch = row / output.Height();
		CFloatHandle outputDataPtr = outputData + batch * output.ObjectSize();
		const int outputRow = row % output.Height();

		if( freeTermData != 0 ) {
			// Set the free term
			setVectorToMatrixRows( outputDataPtr + outputRow * output.Width() * vectorSize, output.Width(),
				vectorSize, *freeTermData );
		} else {
			vectorFill( outputDataPtr + outputRow * output.Width() * vectorSize, 0, output.Width() * vectorSize );
		}

		// Iterate through the filter top positions, starting to apply the filter once we intersect with the current row
		int topPosMinVal = max( outputRow - totalFilterHeight + 1, -desc.PaddingHeight );
		int topPosMaxVal = min( outputRow, output.Height() + desc.PaddingHeight - totalFilterHeight );
		for( int topPos = topPosMinVal; topPos <= topPosMaxVal; topPos++ ) {
			if( ( topPos + desc.PaddingHeight ) % desc.StrideHeight != 0 ) {
				// This position couldn't have been the filter top row
				continue;
			}
			if( ( outputRow - topPos ) % desc.DilationHeight != 0 ) {
				// The filter that starts here doesn't intersect with the current row
				continue;
			}
			const int filterRow = ( outputRow - topPos ) / desc.DilationHeight; // the current filter row

			int sourceVPos = ( topPos + desc.PaddingHeight ) / desc.StrideHeight;
			int sourceHPos = 0;
			// Iterate through the filter left positions
			for( int leftPos = -desc.PaddingWidth; leftPos + totalFilterWidth <= output.Width() + desc.PaddingWidth;
				leftPos += desc.StrideWidth )
			{
				// The pointer to the filter data at (topPos, leftPos) position
				CConstFloatHandle tempData = temp
					+ ( batch * source.Height() * source.Width() + sourceVPos * source.Width() + sourceHPos ) * filter.ObjectSize();

				// Apply the filter row starting at (topPos, leftPos) to the current row
				for( int filterColumn = 0; filterColumn < filter.Width(); filterColumn++ ) {
					const int outputColumn = leftPos + filterColumn * desc.DilationWidth;
					if( 0 <= outputColumn && outputColumn < output.Width() ) {
						float* outputVector = GetRaw(outputDataPtr) + ( outputRow * output.Width() + outputColumn ) * vectorSize;
						const float* tempVector = GetRaw(tempData) + ( filterRow * filter.Width() + filterColumn ) * vectorSize;
						vectorAdd( outputVector, tempVector, outputVector, vectorSize );
					}
				}
				sourceHPos++;
			} 
		}
	}
}

void CCpuMathEngine::blobConvolutionBackwardAlgo1( const CCpuConvolutionDesc& desc, const CFloatHandle& sourceData,
	const CFloatHandle& filterData, const CFloatHandle* freeTerm, const CFloatHandle& resultData )
{
	const CBlobDesc& source = desc.Result;
	const CBlobDesc& filter = desc.Filter;
	const CBlobDesc& result = desc.Source;

	// Transpose the filter for easier calculations
	// The filter depth represented by channels
	const int filterForwardGeometricalSize = filter.Height() * filter.Width() * filter.Depth() * filter.Channels();
	const int filterForwardChannelsCount = filter.BatchWidth();
	const int filterForwardDataSize = filterForwardGeometricalSize * filterForwardChannelsCount;
	CFloatHandleVar filterForward( mathEngine(), filterForwardDataSize );

	TransposeMatrix( 1, filterData, filter.BatchWidth(), 1, filter.ObjectSize(), 1, filterForward.GetHandle(),
		filterForwardDataSize );

	// The results of inverse filter application
	const int tempHeight = source.ObjectCount() * source.Height() * source.Width();
	const int tempWidth = filterForwardGeometricalSize;
	const int tempDataSize = tempHeight * tempWidth;
	CFloatHandleVar temp( mathEngine(), tempDataSize );

	const int curThreadCount = IsOmpRelevant( result.ObjectCount() * result.Height(),
		static_cast<int64_t>( source.BlobSize() ) * filter.BlobSize() ) ? threadCount : 1;

	NEOML_OMP_NUM_THREADS( curThreadCount )
	{
		// Step 1: multiply the input and filter matrices
		int inputStart;
		int inputCount;
		if( OmpGetTaskIndexAndCount( tempHeight, inputStart, inputCount ) ) {
			multiplyMatrixByTransposedMatrix( sourceData + inputStart * filterForwardChannelsCount,
				inputCount, filterForwardChannelsCount, filterForwardChannelsCount,
				filterForward.GetHandle(), filterForwardGeometricalSize, filterForwardChannelsCount,
				temp.GetHandle() + inputStart * tempWidth, filterForwardGeometricalSize, inputCount * tempWidth );
		}

		if( curThreadCount > 1 ) {
			#pragma omp barrier
		}

		// Step 2: add the subvectors from the resulting matrix to the required positions in the output
		if( desc.DilationHeight > 1 || desc.DilationWidth > 1 ) {
			backwardDilationConvolutionAddFilterToOutput( desc, temp.GetHandle(), freeTerm, resultData );
		} else {
			backwardConvolutionAddFilterToOutput( desc, temp.GetHandle(), freeTerm, resultData );
		}
	}
}

// Creates a temporary outputDiff blob using the #2 algorithm
static void createTempBlobsLearnAlgo2( const CBlobDesc& outputDiff, const CBlobDesc& filter, CBlobDesc& result )
{
	result = outputDiff;
	result.SetDimSize( BD_ListSize, 1 );
	result.SetDimSize( BD_BatchLength, 1 );
	result.SetDimSize( BD_BatchWidth, outputDiff.ObjectCount() + 1 );
	result.SetDimSize( BD_Width, outputDiff.Width() + filter.Width() - 1 );
}

// Fills the temporary outputDiff blob using the #2 algorithm
void CCpuMathEngine::fillTempBlobsForLearnAlgo2( const CCpuConvolutionDesc& desc, const CFloatHandle& outputDiffData,
	const CBlobDesc& tempBlob, const CFloatHandle& tempHandle )
{
	const CBlobDesc& outputDiff = desc.Result;

	VectorFill( tempHandle, 0, tempBlob.BlobSize());

	int outputDiffRowSize = outputDiff.Width() * outputDiff.Depth() * outputDiff.Channels();
	int tempDiffRowSize = tempBlob.ObjectSize() / tempBlob.Height();

	const int batchSize = outputDiff.ObjectCount();

	const float* outputDiffDataPtr = GetRaw(outputDiffData);
	for(int j = 0; j < batchSize; ++j) {
		float* tempDiff = GetRaw(tempHandle) + (j + 1) * tempBlob.ObjectSize();
		for(int h = 0; h < outputDiff.Height(); ++h) {
			dataCopy(tempDiff, outputDiffDataPtr, outputDiffRowSize);
			outputDiffDataPtr += outputDiffRowSize;
			tempDiff += tempDiffRowSize;
		}
	}
}

void CCpuMathEngine::blobConvolutionBackwardAlgo2( const CCpuConvolutionDesc& desc, const CFloatHandle& outputDiffData,
	const CFloatHandle& filterData, const CFloatHandle* freeTermData, const CFloatHandle& inputDiffData )
{
	ASSERT_EXPR( desc.StrideHeight == 1 );
	ASSERT_EXPR( desc.StrideWidth == 1 );
	ASSERT_EXPR( desc.PaddingHeight == 0 );
	ASSERT_EXPR( desc.PaddingWidth == 0 );
	ASSERT_EXPR( desc.DilationHeight == 1 );
	ASSERT_EXPR( desc.DilationWidth == 1 );

	const CBlobDesc& filter = desc.Filter;
	const CBlobDesc& inputDiff = desc.Source;
	const CBlobDesc& outputDiff = desc.Result;

	CBlobDesc tempBlobDesc( CT_Float );
	createTempBlobsLearnAlgo2( outputDiff, filter, tempBlobDesc );
	CFloatHandleStackVar tempBlobForLearn( mathEngine(), tempBlobDesc.BlobSize() );
	fillTempBlobsForLearnAlgo2( desc, outputDiffData, tempBlobDesc, tempBlobForLearn.GetHandle() );

	// Repack the filter: switch batch, channels, height and reorder the filter rows backward, end to start
	const int tempFilterObjectSize = filter.Depth() * filter.Channels() * filter.Width() * filter.BatchWidth();
	const int tempFilterDataSize = tempFilterObjectSize * filter.Height();
	CFloatHandleVar tempFilter( mathEngine(), tempFilterDataSize );

	CConstFloatHandle filterDataPtr = filterData;
	for(int b = 0; b < filter.BatchWidth(); ++b) {
		for(int j = 0; j < filter.Height(); ++j) {
			for(int i = 0; i < filter.Width(); ++i) {
				for(int k = 0; k < filter.Depth(); ++k) {
					for(int l = 0; l < filter.Channels(); ++l) {
						int tempFilterPos = ( ( k * filter.Channels() + l ) * filter.Width() + filter.Width() - i - 1 ) * filter.BatchWidth() + b;
						dataCopy(GetRaw(tempFilter.GetHandle()) + j * tempFilterObjectSize + tempFilterPos, GetRaw(filterDataPtr), 1);
						filterDataPtr++;
					}
				}
			}
		}
	}

	const int batchSize = outputDiff.ObjectCount();

	const int curThreadCount = IsOmpRelevant(batchSize) ? threadCount : 1;

	NEOML_OMP_FOR_NUM_THREADS( curThreadCount )
	for( int j = 0; j < batchSize; j++ ) {
		CFloatHandle inputDiffStart = inputDiffData + j * inputDiff.ObjectSize();
		if( freeTermData != 0 ) {
			setVectorToMatrixRows( inputDiffStart, inputDiff.Height() * inputDiff.Width(),
				inputDiff.Depth() * inputDiff.Channels(), *freeTermData );
		} else {
			vectorFill(inputDiffStart, 0, inputDiff.ObjectSize());
		}
		const float* filterStart = GetRaw(tempFilter.GetHandle());
		for(int h = 0; h < filter.Height(); ++h) {
			for(int w = 0; w < filter.Width(); ++w) {
				float* inputDiffMatrix = GetRaw(inputDiffStart) + w * inputDiff.Depth() * inputDiff.Channels();
				const float* outputDiffMatrix = GetRaw( tempBlobForLearn.GetHandle() ) + (j + 1) * tempBlobDesc.ObjectSize()
					+ (w - filter.Width() + 1) * tempBlobDesc.Depth() * tempBlobDesc.Channels();
				int outputDiffHeight = (tempBlobDesc.Height() * tempBlobDesc.Width()
					+ filter.Width() - w - 1) / filter.Width();
				int outputDiffWidth = filter.Width() * tempBlobDesc.Depth() * tempBlobDesc.Channels();
				multiplyMatrixByTransposedMatrixAndAdd( outputDiffMatrix, outputDiffHeight, outputDiffWidth, outputDiffWidth,
					filterStart, filter.Depth() * filter.Channels(), filter.Width() * filter.BatchWidth(),
					inputDiffMatrix, filter.Width() * inputDiff.Depth() * inputDiff.Channels() );
			}

			inputDiffStart += inputDiff.Width() * inputDiff.Depth() * inputDiff.Channels();
			filterStart += tempFilterObjectSize;
		}
	}
}

void CCpuMathEngine::BlobConvolutionBackward( const CConvolutionDesc& convDesc, const CFloatHandle& outputDiffData,
	const CFloatHandle& filter, const CFloatHandle* freeTerm, const CFloatHandle& inputDiffData )
{
	const CCpuConvolutionDesc& desc = static_cast<const CCpuConvolutionDesc&>( convDesc );

	switch( desc.BackwardAlgo ) {
		case CA_1:
			blobConvolutionBackwardAlgo1( desc, outputDiffData, filter, freeTerm, inputDiffData );
			break;
		case CA_2:
			blobConvolutionBackwardAlgo2( desc, outputDiffData, filter, freeTerm, inputDiffData );
			break;
		case CA_1x1:
			{
				bool needsFlatten = desc.Filter.Depth() != 1;

				C3dConvolutionDesc* blob3dConvDesc = InitBlob3dConvolution( needsFlatten ? flatten( desc.Source ) : desc.Source, 0, 0, 0,
					desc.StrideHeight, desc.StrideWidth, 1, needsFlatten ? flatten( desc.Filter ) : desc.Filter, desc.Result );
				Blob3dConvolutionBackward( *blob3dConvDesc, outputDiffData, filter, freeTerm, inputDiffData );
				delete blob3dConvDesc;
				break;
			}
		default:
			ASSERT_EXPR(false);
	}
}

void CCpuMathEngine::blobConvolutionLearnAlgo1( const CCpuConvolutionDesc& desc,
	const CFloatHandle& inputData, const CFloatHandle& outputDiffData, const CFloatHandle& filterDiffData,
	const CFloatHandle* freeTermDiffData, bool isFreeTermDiffFromInput )
{
	const CBlobDesc& input = desc.Source;
	const CBlobDesc& filterDiff = desc.Filter;
	const CBlobDesc& outputDiff = desc.Result;

	ASSERT_EXPR( filterDiff.Depth() == input.Depth() );
	ASSERT_EXPR( filterDiff.Channels() == input.Channels() );

	const int objectCount = outputDiff.ObjectCount();
	const int freeTermDiffSize = isFreeTermDiffFromInput ? filterDiff.Channels() : filterDiff.ObjectCount();

	const int curThreadCount = IsOmpRelevant(objectCount) ? threadCount : 1;

	COmpPrivate2DData outputDiffTrans( curThreadCount, mathEngine(), outputDiff.Width() * outputDiff.Height(),
		outputDiff.Depth() * outputDiff.Channels() );
	COmpPrivate2DData tempBlobHolder( curThreadCount, mathEngine(), outputDiff.Width() * outputDiff.Height(),
		filterDiff.Height() * filterDiff.Width() * input.Depth() * input.Channels() );
	COmpPrivate1DData outputTemp( curThreadCount, mathEngine(), filterDiff.BlobSize() );
	COmpReduction1DData filterDiffItem( mathEngine(), filterDiffData, filterDiff.BlobSize() );
	COmpReduction<COmpReduction1DData> filterDiffReduction( curThreadCount, filterDiffItem );

	unique_ptr<COmpReduction1DData> freeTermDiffItem( nullptr );
	unique_ptr<COmpReduction<COmpReduction1DData>> freeTermDiffReduction( nullptr );

	if( freeTermDiffData != 0 ) {
		freeTermDiffItem.reset( new COmpReduction1DData( mathEngine(), *freeTermDiffData, freeTermDiffSize ) );
		freeTermDiffReduction.reset( new COmpReduction<COmpReduction1DData>( curThreadCount, *freeTermDiffItem ) );
	}

	NEOML_OMP_FOR_NUM_THREADS( curThreadCount )
	for(int b = 0; b < objectCount; ++b) {
		if( desc.DilationHeight > 1 || desc.DilationWidth > 1 ) {
			createDilationTemporaryBlob( desc, inputData, b, 0, outputDiff.Width(), tempBlobHolder.GetPrivateData() );
		} else {
			createTemporaryBlob( desc, inputData, b, 0, outputDiff.Width(), tempBlobHolder.GetPrivateData() );
		}

		transposeMatrixImpl<float>( 1, outputDiffData + b * outputDiff.ObjectSize(),
			outputDiff.Height(), 1, outputDiff.Width(), outputDiff.Depth() * outputDiff.Channels(),
			outputDiffTrans.GetPrivateData(), outputDiffTrans.GetDataSize() );

		// Calculate diffs
		multiplyTransposedMatrixByMatrix( outputDiffTrans.GetPrivateData(),
			outputDiffTrans.GetHeight(), outputDiffTrans.GetWidth(),
			tempBlobHolder.GetPrivateData(), tempBlobHolder.GetWidth(),
			outputTemp.GetPrivateData(), outputTemp.GetDataSize() );

		vectorAdd( GetRaw(filterDiffReduction.GetPrivate().Data), GetRaw(outputTemp.GetPrivateData()),
			GetRaw(filterDiffReduction.GetPrivate().Data), filterDiff.BlobSize() );

		if( freeTermDiffData != 0 ) {
			// Train the free term (add diff to the accumulating data)
			CConstFloatHandle diffData;
			int diffDataHeight;
			int diffDataWidth;

			if( isFreeTermDiffFromInput ) {
				diffData = inputData + b * input.ObjectSize();
				diffDataHeight = input.Height();
				diffDataWidth = input.Width();
			} else {
				diffData = outputDiffTrans.GetPrivateData();
				diffDataHeight = outputDiff.Width();
				diffDataWidth = outputDiff.Height();
			}
			for( int j = 0; j < diffDataHeight; ++j ) {
				for( int k = 0; k < diffDataWidth; ++k ) {
					vectorAdd( GetRaw(freeTermDiffReduction->GetPrivate().Data), GetRaw(diffData),
						GetRaw(freeTermDiffReduction->GetPrivate().Data), freeTermDiffReduction->GetPrivate().Size );
					diffData += freeTermDiffReduction->GetPrivate().Size;
				}
			}
		}
	}

	if( freeTermDiffData != 0 ) {
		freeTermDiffReduction->Reduce();
	}
	filterDiffReduction.Reduce();
}

void CCpuMathEngine::blobConvolutionLearnAlgo2( const CCpuConvolutionDesc& desc, const CFloatHandle& inputData,
	const CFloatHandle& outputDiffData, const CFloatHandle& filterDiffData, const CFloatHandle* freeTermDiffData,
	bool isFreeTermDiffFromInput )
{
	const CBlobDesc& input = desc.Source;
	const CBlobDesc& outputDiff = desc.Result;
	const CBlobDesc& filterDiff = desc.Filter;

	ASSERT_EXPR( desc.StrideHeight == 1 );
	ASSERT_EXPR( desc.StrideWidth == 1 );
	ASSERT_EXPR( desc.PaddingHeight == 0 );
	ASSERT_EXPR( desc.PaddingWidth == 0 );
	ASSERT_EXPR( desc.DilationHeight == 1 );
	ASSERT_EXPR( desc.DilationWidth == 1 );

	CBlobDesc tempBlobDesc( CT_Float );
	createTempBlobsLearnAlgo2( outputDiff, filterDiff, tempBlobDesc );
	CFloatHandleStackVar tempBlobForLearn( mathEngine(), tempBlobDesc.BlobSize() );
	fillTempBlobsForLearnAlgo2( desc, outputDiffData, tempBlobDesc, tempBlobForLearn.GetHandle() );

	const int objectCount = outputDiff.ObjectCount();
	const int curThreadCount = IsOmpRelevant(objectCount) ? threadCount : 1;
	const int freeTermDiffSize = isFreeTermDiffFromInput ? filterDiff.Channels() : filterDiff.ObjectCount();

	COmpReduction1DData filterDiffItem( mathEngine(), filterDiffData, filterDiff.BlobSize() );
	COmpReduction<COmpReduction1DData> filterDiffReduction( curThreadCount, filterDiffItem );

	unique_ptr<COmpReduction1DData> freeTermDiffItem( nullptr );
	unique_ptr<COmpReduction<COmpReduction1DData>> freeTermDiffReduction( nullptr );

	if( freeTermDiffData != 0 ) {
		freeTermDiffItem.reset( new COmpReduction1DData( mathEngine(), *freeTermDiffData, freeTermDiffSize ) );
		freeTermDiffReduction.reset( new COmpReduction<COmpReduction1DData>( curThreadCount, *freeTermDiffItem ) );
	}

	NEOML_OMP_FOR_NUM_THREADS( curThreadCount )
	for( int j = 0; j < objectCount; ++j ) {
		// filter diff
		CFloatHandle filterMatrix = filterDiffReduction.GetPrivate().Data;
		for( int h = 0; h < filterDiff.Height(); ++h ) {
			for( int w = 0; w < filterDiff.Width(); ++w, filterMatrix += filterDiff.Depth() * filterDiff.Channels() ) {
				int matrixHeight = ( input.Height() - filterDiff.Height() + 1 ) * input.Width() - w;
				CConstFloatHandle inputMatrix = inputData + ((j * input.Height() + h) * input.Width() + w) * input.Depth() * input.Channels();
				multiplyTransposedMatrixByMatrixAndAdd(tempBlobForLearn.GetHandle() + (j + 1) * tempBlobDesc.ObjectSize(),
					matrixHeight,
					tempBlobDesc.Depth() * tempBlobDesc.Channels(),
					tempBlobDesc.Depth() * tempBlobDesc.Channels(),
					inputMatrix,
					input.Depth() * input.Channels(),
					input.Depth() * input.Channels(),
					filterMatrix, filterDiff.ObjectSize(),
					filterDiff.ObjectSize() * (filterDiff.BatchWidth() - 1)
						+ (filterDiff.Height() - h) * filterDiff.Width() *
							filterDiff.Depth() * filterDiff.Channels()
							- w * filterDiff.Depth() * filterDiff.Channels());
			}
		}

		if( freeTermDiffData != 0 ) {
			// freeTerm diff
			// Train free term (add diff to the accumulating data)
			CConstFloatHandle diffData;
			int diffDataHeight;
			int diffDataWidth;

			if( isFreeTermDiffFromInput ) {
				diffData = inputData + j * input.ObjectSize();
				diffDataHeight = input.Height();
				diffDataWidth = input.Width();
			} else {
				diffData = outputDiffData + j * outputDiff.ObjectSize();
				diffDataHeight = outputDiff.Height();
				diffDataWidth = outputDiff.Width();
			}
			for( int m = 0; m < diffDataHeight; ++m ) {
				for( int k = 0; k < diffDataWidth; ++k ) {
					vectorAdd( GetRaw(freeTermDiffReduction->GetPrivate().Data), GetRaw(diffData),
						GetRaw(freeTermDiffReduction->GetPrivate().Data), freeTermDiffReduction->GetPrivate().Size );
					diffData += freeTermDiffReduction->GetPrivate().Size;
				}
			}
		}
	}

	filterDiffReduction.Reduce();

	if( freeTermDiffData != 0 ) {
		freeTermDiffReduction->Reduce();
	}
}

void CCpuMathEngine::BlobConvolutionLearnAdd( const CConvolutionDesc& convDesc, const CFloatHandle& input,
	const CFloatHandle& outputDiff, const CFloatHandle& filterDiff, const CFloatHandle* freeTermDiff, bool isFreeTermDiffFromInput )
{
	const CCpuConvolutionDesc& desc = static_cast<const CCpuConvolutionDesc&>( convDesc );

	switch( desc.BackwardAlgo ) {
		case CA_1:
			blobConvolutionLearnAlgo1( desc, input, outputDiff, filterDiff, freeTermDiff, isFreeTermDiffFromInput );
			break;
		case CA_2:
			blobConvolutionLearnAlgo2( desc, input, outputDiff, filterDiff, freeTermDiff, isFreeTermDiffFromInput );
			break;
		case CA_1x1:
			{
				bool needsFlatten = desc.Filter.Depth() != 1;

				C3dConvolutionDesc* blob3dConvDesc = InitBlob3dConvolution( needsFlatten ? flatten( desc.Source ) : desc.Source , 0, 0, 0,
					desc.StrideHeight, desc.StrideWidth, 1, needsFlatten ? flatten( desc.Filter ) : desc.Filter, desc.Result );
				Blob3dConvolutionLearnAdd( *blob3dConvDesc, input, outputDiff, filterDiff, freeTermDiff, true );
				delete blob3dConvDesc;
				break;
			}
		default:
			ASSERT_EXPR( false );
	}
}

//------------------------------------------------------------------------------------------------------------

CChannelwiseConvolutionDesc* CCpuMathEngine::InitBlobChannelwiseConvolution( const CBlobDesc& source,
	int paddingHeight, int paddingWidth, int strideHeight, int strideWidth,
	const CBlobDesc& filter, const CBlobDesc* freeTerm, const CBlobDesc& result )
{
	ASSERT_EXPR(source.Depth() == 1);
	ASSERT_EXPR(filter.Height() > paddingHeight);
	ASSERT_EXPR(filter.Height() <= source.Height() + 2 * paddingHeight);
	ASSERT_EXPR(filter.Width() > paddingWidth);
	ASSERT_EXPR(filter.Width() <= source.Width() + 2 * paddingWidth);
	ASSERT_EXPR(filter.ObjectCount() == 1);
	ASSERT_EXPR(filter.Channels() == source.Channels());
	ASSERT_EXPR(freeTerm == 0 || freeTerm->BlobSize() == filter.Channels());
	ASSERT_EXPR(result.BatchLength() == source.BatchLength());
	ASSERT_EXPR(result.BatchWidth() == source.BatchWidth());
	ASSERT_EXPR(result.Depth() == 1);
	ASSERT_EXPR(result.Channels() == source.Channels());
	const int expectedOutputHeight = (source.Height() - filter.Height() + 2 * paddingHeight) / strideHeight + 1;
	const int expectedOutputWidth = (source.Width() - filter.Width() + 2 * paddingWidth) / strideWidth + 1;
	ASSERT_EXPR(result.Height() == expectedOutputHeight);
	ASSERT_EXPR(result.Width() == expectedOutputWidth);

	CCommonChannelwiseConvolutionDesc* desc = new CCommonChannelwiseConvolutionDesc( paddingHeight, paddingWidth,
		strideHeight, strideWidth, source, filter, result );
	return desc;
}

void CCpuMathEngine::BlobChannelwiseConvolutionBackward( const CChannelwiseConvolutionDesc& convDesc,
	const CFloatHandle& inputDiffData, const CFloatHandle& filterData, const CFloatHandle& outputDiffData )
{
	const CCommonChannelwiseConvolutionDesc& desc = static_cast<const CCommonChannelwiseConvolutionDesc&>( convDesc );
	const CBlobDesc& input = desc.Result;
	const CBlobDesc& filter = desc.Filter;
	const CBlobDesc& output = desc.Source;

	const int inputGeo = input.Height() * input.Width();
	const int filterGeo = filter.Height() * filter.Width();
	const int inputBatch = input.Channels() * inputGeo;
	const int outputBatch = output.Channels() * output.Height() * output.Width();

	// Transpose the: HWC -> CHW
	CFloatHandleStackVar filterTransposed( mathEngine(), filter.BlobSize() );
	TransposeMatrix(1, filterData, filterGeo, 1, filter.Channels(), 1, filterTransposed, filterTransposed.Size());

	const int curThreadCount = IsOmpRelevant( input.BatchWidth() ) ? threadCount : 1;

	COmpPrivate2DData inputRepacked( curThreadCount, mathEngine(), input.Height() * input.Width(), input.Channels() );
	COmpPrivate1DData temp( curThreadCount, mathEngine(), inputGeo * filterGeo * input.Channels() );
	COmpPrivate2DData outputRepacked( curThreadCount, mathEngine(), output.Height() * output.Width(), output.Channels() );

	NEOML_OMP_FOR_NUM_THREADS( curThreadCount )
	for( int batchIndex = 0; batchIndex < input.BatchWidth(); ++batchIndex ) {
		// Repack HWC -> CHW
		transposeMatrixImpl<float>( 1, inputDiffData + batchIndex * inputBatch,
			inputGeo, 1, input.Channels(), 1, inputRepacked.GetPrivateData(), inputRepacked.GetDataSize() );

		// Multiply the inputRepacked and filter matrices
		batchMultiplyMatrixByTransposedMatrix( inputRepacked.GetWidth(),
			inputRepacked.GetPrivateData(), inputGeo, 1,
			filterTransposed, filterGeo, temp.GetPrivateData(), temp.GetDataSize() );

		// Add the subvectors from the resulting matrix to the required positions in outputRepacked
		for( int step = 0; step < output.Height() * output.Channels(); ++step ) {
			float* outputDataPtr = GetRaw( outputRepacked.GetPrivateData() ) + step * output.Width();
			NeoML::vectorFill( outputDataPtr, 0, output.Width() );

			const int channel = step / output.Height();
			const int row = step % output.Height();
			int inputRowStart = ( row + desc.PaddingHeight - filter.Height() + desc.StrideHeight ) / desc.StrideHeight;
			if( inputRowStart < 0 ) {
				inputRowStart = 0;
			}
			const int filterRowBackStart = row - inputRowStart * desc.StrideHeight + desc.PaddingHeight;
			if( 0 > filterRowBackStart || filterRowBackStart >= filter.Height() ) {
				continue;
			}
			int filterRowBackEnd = filter.Height() + row - output.Height() - desc.PaddingHeight;
			if( filterRowBackEnd < 0 ) {
				filterRowBackEnd = 0;
			}

			int inputRow = inputRowStart;
			for( int filterRow = filterRowBackStart;
				filterRow >= filterRowBackEnd;
				filterRow -= desc.StrideHeight, ++inputRow )
			{
				// The temp blob stores the rows of the filter multiplied by input; add them to the output rows in required positions
				const float* tempRowData = GetRaw( temp.GetPrivateData() ) + ( ( channel * input.Height() + inputRow )
					* input.Width() * filter.Height() + filterRow ) * filter.Width();

				for( int col = -desc.PaddingWidth;
					col <= output.Width() + desc.PaddingWidth - filter.Width();
					col += desc.StrideWidth )
				{

					int tempRowDataShift = 0;
					int toCopy = filter.Width();
					int pos = col;
					if( pos < 0 ) {
						tempRowDataShift = -pos;
						toCopy += pos;
						pos = 0;
					}
					if( pos + toCopy > output.Width() ) {
						toCopy = output.Width() - pos;
					}

					vectorAdd( outputDataPtr + pos, tempRowData + tempRowDataShift, outputDataPtr + pos, toCopy );

					tempRowData += filter.Height() * filter.Width();
				}
			}
		}

		// Repack CHW -> HWC
		transposeMatrixImpl<float>( 1, outputRepacked.GetPrivateData(),
			outputRepacked.GetWidth(), 1, outputRepacked.GetHeight(), 1,
			outputDiffData + batchIndex * outputBatch, outputBatch );
	}
}

void CCpuMathEngine::BlobChannelwiseConvolutionLearnAdd( const CChannelwiseConvolutionDesc& convDesc, const CFloatHandle& inputData,
	const CFloatHandle& outputDiffData, const CFloatHandle& filterDiffData, const CFloatHandle* freeTermDiffData )
{
	const CCommonChannelwiseConvolutionDesc& desc = static_cast<const CCommonChannelwiseConvolutionDesc&>( convDesc );
	const CBlobDesc& outputDiff = desc.Result;
	const CBlobDesc& input = desc.Source;
	const CBlobDesc& filterDiff = desc.Filter;

	const int curThreadCount = IsOmpRelevant( outputDiff.BatchWidth() ) ? threadCount : 1;

	COmpPrivate2DData outputDiffTrans( curThreadCount, mathEngine(), outputDiff.Width() * outputDiff.Height(), input.Channels() );
	COmpPrivate2DData outputDiffTransRepacked( curThreadCount, mathEngine(), outputDiff.Width() * outputDiff.Height(), input.Channels() );
	COmpPrivate2DData tempBlob( curThreadCount, mathEngine(), outputDiff.Width() * outputDiff.Height()
		* filterDiff.Height() * filterDiff.Width(), input.Channels() );
	COmpPrivate1DData tempBlobRepacked( curThreadCount, mathEngine(), outputDiff.Width() * outputDiff.Height()
		* filterDiff.Height() * filterDiff.Width() * input.Channels() );
	COmpPrivate1DData filterTemp( curThreadCount, mathEngine(), filterDiff.Channels() * filterDiff.Height() * filterDiff.Width() ); // a blob with one object diffs

	// The transposed filter diff
	CFloatHandleStackVar filterDiffTransposedHolder( mathEngine(), filterDiff.BlobSize() );
	CBlobDesc filterDiffTransposed( CT_Float );
	filterDiffTransposed.SetDimSize(BD_BatchWidth, filterDiff.Channels());
	filterDiffTransposed.SetDimSize(BD_Height, filterDiff.Height());
	filterDiffTransposed.SetDimSize(BD_Width, filterDiff.Width());
	TransposeMatrix( 1, filterDiffData, filterDiff.Height() * filterDiff.Width(), 1, filterDiff.Channels(), 1,
		filterDiffTransposedHolder.GetHandle(), filterDiffTransposed.BlobSize() );

	COmpReduction1DData filterDiffItem( mathEngine(), filterDiffTransposedHolder.GetHandle(), filterDiffTransposed.BlobSize() );
	COmpReduction<COmpReduction1DData> filterDiffReduction( curThreadCount, filterDiffItem );

	unique_ptr<COmpReduction1DData> freeTermDiffItem( nullptr );
	unique_ptr<COmpReduction<COmpReduction1DData>> freeTermDiffReduction( nullptr );

	if( freeTermDiffData != 0 ) {
		freeTermDiffItem.reset( new COmpReduction1DData( mathEngine(), *freeTermDiffData, filterDiff.Channels() ) );
		freeTermDiffReduction.reset( new COmpReduction<COmpReduction1DData>( curThreadCount, *freeTermDiffItem ) );
	}

	NEOML_OMP_NUM_THREADS( curThreadCount )
	{
		int batchStart;
		int batchCount;
		if( OmpGetTaskIndexAndCount( outputDiff.BatchWidth(), batchStart, batchCount ) ) {
			for( int batchIndex = batchStart; batchIndex < batchStart + batchCount; ++batchIndex ) {
				// Filling the matrix from the windows
				createTemporaryBlob( desc, inputData, batchIndex, 0, outputDiff.Width(), tempBlob.GetPrivateData() );
				// Repack HWC -> CHW
				transposeMatrixImpl<float>( 1, tempBlob.GetPrivateData(), tempBlob.GetHeight(), 1, tempBlob.GetWidth(), 1,
					tempBlobRepacked.GetPrivateData(), tempBlobRepacked.GetDataSize() );

				// Transpose the output blob HWC -> WHC
				transposeMatrixImpl<float>( 1, outputDiffData + batchIndex * outputDiff.ObjectSize(),
					outputDiff.Height(), 1, outputDiff.Width(), outputDiff.Channels(),
					outputDiffTrans.GetPrivateData(), outputDiffTrans.GetDataSize() );
				// Repack HWC -> CHW
				transposeMatrixImpl<float>( 1, outputDiffTrans.GetPrivateData(),
					outputDiffTrans.GetHeight(), 1, outputDiffTrans.GetWidth(), 1,
					outputDiffTransRepacked.GetPrivateData(), outputDiffTransRepacked.GetDataSize() );

				// Multiply matrices
				batchMultiplyTransposedMatrixByMatrix( outputDiffTransRepacked.GetWidth(),
					outputDiffTransRepacked.GetPrivateData(),
					outputDiffTransRepacked.GetHeight(), 1,
					tempBlobRepacked.GetPrivateData(), filterDiff.Height() * filterDiff.Width(),
					filterTemp.GetPrivateData(), filterTemp.GetDataSize() );

				// Update the accumulator
				vectorAdd( GetRaw( filterDiffReduction.GetPrivate().Data ), GetRaw( filterTemp.GetPrivateData() ),
					GetRaw( filterDiffReduction.GetPrivate().Data ), filterTemp.GetDataSize() );

				if( freeTermDiffData != 0 ) {
					// Train the free term (add diffs to the accumulated data)
					sumMatrixColumnsAdd( freeTermDiffReduction->GetPrivate().Data, outputDiffTransRepacked.GetPrivateData(),
						outputDiffTransRepacked.GetWidth(), outputDiffTransRepacked.GetHeight() );
				}
			}
		}
	}

	if( freeTermDiffData != 0 ) {
		freeTermDiffReduction->Reduce();
	}

	filterDiffReduction.Reduce();

	TransposeMatrix( 1, filterDiffTransposedHolder.GetHandle(),
		filterDiff.Channels(), 1, filterDiff.Height() * filterDiff.Width(), 1, filterDiffData, filterDiff.BlobSize() );
}

} // namespace NeoML
