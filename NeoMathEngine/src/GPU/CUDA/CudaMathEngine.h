/* Copyright © 2017-2023 ABBYY

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

#pragma once

#include <NeoMathEngine/NeoMathEngineDefs.h>

#ifdef NEOML_USE_CUDA

#include <NeoMathEngine/NeoMathEngine.h>
#include <DllLoader.h>
#include <RawMemoryManager.h>
#include <cusparse.h>
#include <cublas.h>
#include <mutex>
#include <memory>
#include <PerformanceCountersDefault.h>
#include <CudaMathEngineDnnDistributed.h>

namespace NeoML {

struct CCudaTimeConvolutionDescInternal;
struct CCudaRleConvolutionDesc;
struct CCudaConvolutionDescInternal;
struct CCuda3dConvolutionDescInternal;
struct CCusparse;
struct CCublas;
struct CCudaDevice;
class CDeviceStackAllocator;
class CHostStackAllocator;
class CMemoryPool;

// CUDA math engine
class CCudaMathEngine : public IMathEngine, public IRawMemoryManager {
public:
	CCudaMathEngine( const CCusparse* cusparse, const CCublas* cublas, std::unique_ptr<CCudaDevice>& device, int flags = 0 );
	~CCudaMathEngine() override;

	// IMathEngine interface methods
	TMathEngineType GetType() const override { return MET_Cuda; }
	void GetMathEngineInfo( CMathEngineInfo& info ) const override;
	void SetReuseMemoryMode( bool enable ) override;
	CMemoryHandle HeapAlloc( size_t count ) override;
	void HeapFree( const CMemoryHandle& handle ) override;
	CMemoryHandle StackAlloc( size_t count ) override;
	void StackFree( const CMemoryHandle& handle ) override;
	size_t GetFreeMemorySize() const override;
	size_t GetPeakMemoryUsage() const override;
	size_t GetMemoryInPools() const override;
	void CleanUp() override;
	void* GetBuffer( const CMemoryHandle& handle, size_t pos, size_t size, bool exchange ) override;
	void ReleaseBuffer( const CMemoryHandle& handle, void* ptr, bool exchange ) override;
	void DataExchangeRaw( const CMemoryHandle& handle, const void* data, size_t size ) override;
	void DataExchangeRaw( void* data, const CMemoryHandle& handle, size_t size ) override;
	CMemoryHandle CopyFrom( const CMemoryHandle& handle, size_t size ) override;

	// IVectorMathematicsEngine interface methods
	void VectorFill( const CFloatHandle& result, float value, int vectorSize ) override;
	void VectorFill( const CIntHandle& result, int value, int vectorSize ) override;
	void VectorFill( const CFloatHandle& result, int vectorSize, const CConstFloatHandle& value ) override;
	void VectorFill( const CIntHandle& result, int vectorSize, const CConstIntHandle& value ) override;
	void VectorConvert( const CConstFloatHandle& from, const CIntHandle& to, int vectorSize ) override;
	void VectorConvert( const CConstIntHandle& from, const CFloatHandle& to, int vectorSize ) override;
	void VectorFillBernoulli( const CFloatHandle& result, float p, int vectorSize, float value, int seed ) override;
	void FilterSmallValues( const CFloatHandle& data, int dataSize, float threshold ) override;
	void VectorCopy( const CFloatHandle& first, const CConstFloatHandle& second, int vectorSize ) override;
	void VectorCopy( const CIntHandle& first, const CConstIntHandle& second, int vectorSize ) override;
	void BroadcastCopy( const CIntHandle& toHandle, const CConstIntHandle& fromHandle,
		const CBlobDesc& toDesc, const CBlobDesc& fromDesc, int additionalWidth ) override;
	void BroadcastCopy( const CFloatHandle& toHandle, const CConstFloatHandle& fromHandle,
		const CBlobDesc& toDesc, const CBlobDesc& fromDesc, int additionalWidth ) override;
	void VectorSum( const CConstFloatHandle& firstHandle, int vectorSize, const CFloatHandle& resultHandle ) override;
	void VectorSumAdd( const CConstFloatHandle& firstHandle, int vectorSize, const CFloatHandle& resultHandle ) override;
	void VectorSumAlongDimension( const CConstFloatHandle& firstHandle, int precedingDimension, int dimension,
		int followingDimension, const CFloatHandle& resultHandle ) override;
	void VectorCumSumAlongDimension( const CConstFloatHandle& firstHandle, int precedingDimension, int dimension,
		int followingDimension, const CFloatHandle& resultHandle, bool reverse ) override;
	void VectorCumSumAlongDimension( const CConstIntHandle& firstHandle, int precedingDimension, int dimension,
		int followingDimension, const CIntHandle& resultHandle, bool reverse ) override;
	void VectorSumAlongDimensionDiag( const CConstFloatHandle& firstHandle, int precedingDimension, int dimension,
		int followingDimension, const CFloatHandle& resultHandle ) override;
	void VectorCumSumAlongDimensionDiag( const CConstFloatHandle& firstHandle, int precedingDimension, int dimension,
		int followingDimension, const CFloatHandle& resultHandle ) override;
	void VectorEqual( const CConstIntHandle& firstHandle, const CConstIntHandle& secondHandle,
		const CFloatHandle& resultHandle, int vectorSize ) override;
	void VectorEqualValue( const CConstIntHandle& firstHandle,
		const CFloatHandle& resultHandle, int vectorSize, const CConstIntHandle& valueHandle ) override;
	void VectorMax( const CConstFloatHandle& firstHandle, float secondValue, const CFloatHandle& resultHandle,
		int vectorSize ) override;
	void VectorMaxDiff( const CConstFloatHandle& firstHandle, float secondValue, const CFloatHandle& gradHandle,
		int gradHeight, int gradWidth ) override;
	void VectorELU( const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle,
		int vectorSize, const CConstFloatHandle& alpha ) override;
	void VectorELUDiff( const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
		const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& alpha ) override;
	void VectorELUDiffOp( const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
		const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& alpha ) override;
	void VectorReLU( const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize,
		const CConstFloatHandle& upperThresholdHandle ) override;
	void VectorReLUDiff( const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
		const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& upperThresholdHandle ) override;
	void VectorReLUDiffOp( const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
		const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& upperThresholdHandle ) override;
	void VectorLeakyReLU( const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle,
		int vectorSize, const CConstFloatHandle& alpha ) override;
	void VectorLeakyReLUDiff( const CConstFloatHandle& firstHandle,
		const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle,
		int vectorSize, const CConstFloatHandle& alpha ) override;
	void VectorLeakyReLUDiffOp( const CConstFloatHandle& firstHandle,
		const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle,
		int vectorSize, const CConstFloatHandle& alpha ) override;
	void VectorHSwish( const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle,
		int vectorSize ) override;
	void VectorHSwishDiff( const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
		const CFloatHandle& resultHandle, int vectorSize ) override;
	void VectorEltwiseMax( const CConstFloatHandle& firstHandle,
		const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle, int vectorSize ) override;
	void VectorEltwiseMin( const CConstFloatHandle& firstHandle,
		const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle, int vectorSize ) override;
	void VectorAbs( const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize ) override;
	void VectorAbsDiff( const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
		const CFloatHandle& resultHandle, int vectorSize ) override;
	void VectorAbsDiff( const CConstFloatHandle& sourceGradHandle, int gradHeight, int gradWidth,
		const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle ) override;
	void VectorHinge( const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize ) override;
	void VectorHingeDiff( const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
		const CFloatHandle& resultHandle, int vectorSize ) override;
	void VectorSquaredHinge( const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize ) override;
	void VectorSquaredHingeDiff( const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
		const CFloatHandle& resultHandle, int vectorSize ) override;
	void VectorHuber( const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize ) override;
	void VectorHuberDerivative( const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize ) override;
	void VectorHardTanh( const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize ) override;
	void VectorHardTanhDiff( const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
		const CFloatHandle& resultHandle, int vectorSize ) override;
	void VectorHardTanhDiffOp( const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
		const CFloatHandle& resultHandle, int vectorSize ) override;
	void VectorHardSigmoid( const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize,
		const CConstFloatHandle& slopeHandle, const CConstFloatHandle& biasHandle ) override;
	void VectorHardSigmoidDiff( const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
		const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& slopeHandle, const CConstFloatHandle& biasHandle ) override;
	void VectorHardSigmoidDiffOp( const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
		const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& slopeHandle, const CConstFloatHandle& biasHandle ) override;
	void VectorNeg( const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize ) override;
	void VectorExp( const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize ) override;
	void VectorLog( const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle,
		int vectorSize ) override;
	void VectorLogDiff( const CConstFloatHandle& sourceGradHandle, int sourceGradHeight, int sourceGradWidth,
		const CConstFloatHandle& valueHandle, const CFloatHandle& resultHandle ) override;
	void VectorNegLog( const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize ) override;
	void VectorErf( const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize ) override;
	void VectorBernulliKLDerivative( const CConstFloatHandle& estimationHandle,
		const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& target ) override;
	void VectorAdd( const CConstFloatHandle& firstHandle,
		const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle, int vectorSize ) override;
	void VectorAdd( const CConstIntHandle& firstHandle,
		const CConstIntHandle& secondHandle, const CIntHandle& resultHandle, int vectorSize ) override;
	void VectorAddValue( const CConstFloatHandle& firstHandle,
		const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& addition ) override;
	void VectorAddValue( const CConstIntHandle& firstHandle,
		const CIntHandle& resultHandle, int vectorSize, const CConstIntHandle& addition ) override;
	void VectorSub( const CConstIntHandle& firstHandle,
		const CConstIntHandle& secondHandle, const CIntHandle& resultHandle, int vectorSize ) override;
	void VectorSub( const CConstFloatHandle& firstHandle,
		const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle, int vectorSize ) override;
	void VectorSub( const CConstFloatHandle& firstHandle, float second, const CFloatHandle& resultHandle,
		int vectorSize ) override;
	void VectorSub( float first, const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle,
		int vectorSize ) override;
	void VectorMultiplyAndAdd( const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
		const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& multHandle ) override;
	void VectorMultiplyAndSub( const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
		const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& multHandle ) override;
	void VectorMultiply( const CConstFloatHandle& firstHandle,
		const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& multiplierHandle ) override;
	void VectorMultiply( const CConstIntHandle& firstHandle,
		const CIntHandle& resultHandle, int vectorSize, const CConstIntHandle& multiplierHandle ) override;
	void VectorNegMultiply( const CConstFloatHandle& firstHandle,
		const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& multiplierHandle ) override;
	void VectorEltwiseMultiply( const CConstIntHandle& firstHandle,
		const CConstIntHandle& secondHandle, const CIntHandle& resultHandle, int vectorSize ) override;
	void VectorEltwiseMultiply( const CConstFloatHandle& firstHandle,
		const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle, int vectorSize ) override;
	void VectorEltwiseMultiplyAdd( const CConstFloatHandle& firstHandle,
		const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle, int vectorSize ) override;
	void VectorEltwiseNegMultiply( const CConstFloatHandle& firstHandle,
		const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle, int vectorSize ) override;
	void VectorEltwiseDivide( const CConstIntHandle& firstHandle,
		const CConstIntHandle& secondHandle, const CIntHandle& resultHandle, int vectorSize ) override;
	void VectorEltwiseDivide( const CConstFloatHandle& firstHandle,
		const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle, int vectorSize ) override;
	void VectorEltwisePower( const CConstFloatHandle& firstHandle,
		const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle, int vectorSize ) override;
	void VectorSqrt( const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize ) override;
	void VectorInv( const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize ) override;
	void VectorMinMax( const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize,
		const CConstFloatHandle& minHandle, const CConstFloatHandle& maxHandle ) override;
	void VectorMinMaxDiff( const CConstFloatHandle& sourceGradHandle, int gradHeight, int gradWidth,
		const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle,
		const CConstFloatHandle& minHandle, const CConstFloatHandle& maxHandle ) override;
	void VectorSigmoid( const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize ) override;
	void VectorSigmoidDiff( const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
		const CFloatHandle& resultHandle, int vectorSize ) override;
	void VectorSigmoidDiffOp( const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
		const CFloatHandle& resultHandle, int vectorSize ) override;
	void VectorTanh( const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize ) override;
	void VectorTanhDiff( const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
		const CFloatHandle& resultHandle, int vectorSize ) override;
	void VectorTanhDiffOp( const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
		const CFloatHandle& resultHandle, int vectorSize ) override;
	void VectorPower( float exponent, const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize ) override;
	void VectorPowerDiff( float exponent, const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
		const CFloatHandle& resultHandle, int vectorSize ) override;
	void VectorPowerDiffOp( float exponent, const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
		const CFloatHandle& resultHandle, int vectorSize ) override;
	void VectorL1DiffAdd( const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
		const CFloatHandle& resultHandle, int vectorSize,
		const CConstFloatHandle& hubertThresholdHandle, const CConstFloatHandle& multHandle ) override;
	void VectorDotProduct( const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle, int vectorSize,
		const CFloatHandle& resultHandle ) override;
	void VectorEltwiseNot( const CConstIntHandle& firstHandle, const CIntHandle& resultHandle, int vectorSize ) override;
	void VectorEltwiseNotNegative( const CConstIntHandle& firstHanle, const CFloatHandle& resultHandle, int vectorSize ) override;
	void VectorEltwiseLess( const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
		const CFloatHandle& resultHandle, int vectorSize ) override;
	void VectorEltwiseLess( const CConstFloatHandle& firstHandle, float second, const CFloatHandle& resultHandle, int vectorSize ) override;
	void VectorEltwiseLess( float firstHandle, const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle, int vectorSize ) override;
	void VectorEltwiseLess( const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
		const CIntHandle& resultHandle, int vectorSize ) override;
	void VectorEltwiseLess( const CConstIntHandle& firstHandle, const CConstIntHandle& secondHandle,
		const CIntHandle& resultHandle, int vectorSize ) override;
	void VectorEltwiseEqual( const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
		const CIntHandle& resultHandle, int vectorSize ) override;
	void VectorEltwiseEqual( const CConstIntHandle& firstHandle, const CConstIntHandle& secondHandle,
		const CIntHandle& resultHandle, int vectorSize ) override;
	void VectorEltwiseWhere( const CConstIntHandle& firstHandle, const CConstFloatHandle& secondHandle,
		const CConstFloatHandle& thirdHandle, const CFloatHandle& resultHandle, int vectorSize ) override;
	void VectorEltwiseWhere( const CConstIntHandle& firstHandle, const CConstIntHandle& secondHandle,
		const CConstIntHandle& thirdHandle, const CIntHandle& resultHandle, int vectorSize ) override;
	void VectorFindMaxValueInSet( const CConstFloatHandle* vectors, int vectorCount, const CFloatHandle& resultHandle, int vectorSize ) override;
	void VectorFindMaxValueInSet( const CConstFloatHandle* vectors, int vectorCount, const CFloatHandle& resultHandle,
		const CIntHandle& indexHandle, int vectorSize ) override;
	void VectorSpreadValues( const CConstFloatHandle& sourceHandle, CFloatHandle* vectors, int vectorCount,
		const CConstIntHandle& indexHandle, int vectorSize ) override;
	void VectorTopK( const CConstFloatHandle& first, int firstSize, int k, const CFloatHandle& result, const CIntHandle& indices ) override;
	void VectorTopKDiff( const CConstFloatHandle& sourceGrad, int sourceGradHeight, int sourceGradWidth,
		const CConstIntHandle& indices, int k, const CFloatHandle& resultGrad ) override;

	// IBlasEngine interface methods
	void SetVectorToMatrixRows( const CFloatHandle& resultHandle, int matrixHeight,
		int matrixWidth, const CConstFloatHandle& vectorHandle ) override;
	void AddVectorToMatrixElements( const CFloatHandle& matrix, int height, int width,
		const CConstIntHandle& indices, const CConstFloatHandle& vector ) override;
	void AddVectorToMatrixElements( const CFloatHandle& matrix, int height, int width,
		const CConstIntHandle& rowIndices, const CConstIntHandle& columnIndices,
		const CConstFloatHandle& vector, int vectorSize ) override;
	void AddMatrixElementsToVector( const CConstFloatHandle& matrix, int height, int width,
		const CConstIntHandle& indices, const CFloatHandle& result, int vectorSize ) override;
	void AddMatrixElementsToVector( const CConstFloatHandle& matrix, int height, int width,
		const CConstIntHandle& rowIndices, const CConstIntHandle& columnIndices,
		const CFloatHandle& result, int vectorSize ) override;
	void AddDiagMatrixToMatrix( const CConstFloatHandle& diagMatrix, const CConstFloatHandle& matrix,
		int height, int width, const CFloatHandle& result ) override;
	void AddMatrixElementsToMatrix( const CConstFloatHandle& matrix, int height, int width,
		const CFloatHandle& result, const CConstIntHandle& indices ) override;
	void AddVectorToMatrixRows( int batchSize, const CConstFloatHandle& matrixHandle, const CFloatHandle& resultHandle,
		int matrixHeight, int matrixWidth, const CConstFloatHandle& vectorHandle ) override;
	void AddVectorToMatrixColumns( const CConstFloatHandle& matrixHandle, const CFloatHandle& resultHandle,
		int matrixHeight, int matrixWidth, const CConstFloatHandle& vectorHandle ) override;
	void AddVectorToMatrixColumns( const CConstIntHandle& matrixHandle, const CIntHandle& resultHandle,
		int matrixHeight, int matrixWidth, const CConstIntHandle& vectorHandle ) override;
	void SubVectorFromMatrixColumns( const CConstFloatHandle& matrixHandle, const CFloatHandle& resultHandle,
		int matrixHeight, int matrixWidth, const CConstFloatHandle& vectorHandle ) override;
	void SumMatrixRowsAdd( int batchSize, const CFloatHandle& resultHandle, const CConstFloatHandle& matrixHandle,
		int matrixHeight, int matrixWidth ) override;
	void SumMatrixRows( int batchSize, const CFloatHandle& resultHandle, const CConstFloatHandle& matrixHandle,
		int matrixHeight, int matrixWidth ) override;
	void SumMatrixRows( int batchSize, const CIntHandle& resultHandle, const CConstIntHandle& matrixHandle,
		int matrixHeight, int matrixWidth ) override;
	void SumMatrixColumns( const CFloatHandle& resultHandle, const CConstFloatHandle& matrixHandle,
		int matrixHeight, int matrixWidth ) override;
	void MatrixColumnsEltwiseDivide( const CConstFloatHandle& matrix, int matrixHeight, int matrixWidth,
		const CConstFloatHandle& vector, const CFloatHandle& resultHandle ) override;
	void MatrixLogSumExpByRows( const CConstFloatHandle& matrix, int height, int width, const CFloatHandle& result, int resultSize ) override;
	void MatrixSoftmaxByRows( const CConstFloatHandle& matrix, int height, int width, const CFloatHandle& result ) override;
	void MatrixSoftmaxDiffOpByRows( const CConstFloatHandle& first, const CConstFloatHandle& second,
		int height, int width, const CFloatHandle& result ) override;
	void MatrixSoftmaxByColumns( const CConstFloatHandle& matrix, int height, int width,
		const CFloatHandle& result ) override;
	void MatrixSoftmaxDiffOpByColumns( const CConstFloatHandle& first, const CConstFloatHandle& second,
		int height, int width, const CFloatHandle& result ) override;
	void FindMaxValueInRows( const CConstFloatHandle& matrixHandle, int matrixHeight, int matrixWidth,
		const CFloatHandle& resultHandle, const CIntHandle& columnIndices, int vectorSize ) override;
	void FindMaxValueInRows( const CConstFloatHandle& matrixHandle,
		int matrixHeight, int matrixWidth, const CFloatHandle& resultHandle, int vectorSize ) override;
	void FindMaxValueInColumns( int batchSize, const CConstFloatHandle& matrixHandle, int matrixHeight,
		int matrixWidth, const CFloatHandle& resultHandle, const CIntHandle& rowIndices, int vectorSize ) override;
	void FindMinValueInColumns( const CConstFloatHandle& matrixHandle, int matrixHeight, int matrixWidth,
		const CFloatHandle& resultHandle, const CIntHandle& columnIndices ) override;
	void VectorMultichannelLookupAndCopy( int batchSize, int channelCount, const CConstFloatHandle& inputHandle,
		const CConstFloatHandle* lookupHandles, const CLookupDimension* lookupDimensions, int lookupCount,
		const CFloatHandle& outputHandle, int outputChannels ) override;
	void VectorMultichannelLookupAndCopy( int batchSize, int channelCount, const CConstIntHandle& inputHandle,
		const CConstFloatHandle* lookupHandles, const CLookupDimension* lookupDimensions, int lookupCount,
		const CFloatHandle& outputHandle, int outputChannels ) override;
	void VectorMultichannelLookupAndCopy( int batchSize, int channelCount, const CConstIntHandle& inputHandle,
		const CConstIntHandle* lookupHandles, const CLookupDimension* lookupDimensions, int lookupCount,
		const CIntHandle& outputHandle, int outputChannels ) override;
	void VectorMultichannelLookupAndAddToTable( int batchSize, int channelCount, const CConstFloatHandle& inputHandle,
		const CFloatHandle* lookupHandles, const CLookupDimension* lookupDimensions, int lookupCount,
		const CConstFloatHandle& multHandle, const CConstFloatHandle& matrixHandle, int outputChannels ) override;
	void VectorMultichannelLookupAndAddToTable( int batchSize, int channelCount, const CConstIntHandle& inputHandle,
		const CFloatHandle* lookupHandles, const CLookupDimension* lookupDimensions, int lookupCount,
		const CConstFloatHandle& multHandle, const CConstFloatHandle& matrixHandle, int outputChannels ) override;
	void LookupAndSum( const CConstIntHandle& indicesHandle, int batchSize, int indexCount,
		const CConstFloatHandle& tableHandle, int vectorSize, const CFloatHandle& result ) override;
	void LookupAndAddToTable( const CConstIntHandle& indicesHandle, int batchSize, int indexCount,
		const CConstFloatHandle& additionsHandle, int vectorSize, const CFloatHandle& tableHandle, int vectorCount ) override;
	void EnumBinarization( int batchSize, const CConstFloatHandle& inputHandle, int enumSize,
		const CFloatHandle& resultHandle ) override;
	void EnumBinarization( int batchSize, const CConstIntHandle& inputHandle, int enumSize,
		const CFloatHandle& resultHandle ) override;
	void BitSetBinarization( int batchSize, int bitSetSize,
		const CConstIntHandle& inputHandle, int outputVectorSize, const CFloatHandle& resultHandle ) override;
	void MultiplyLookupMatrixByLookupVector( int batchSize, const CLookupMatrix& matrix,
		const CLookupVector& vector, const CFloatHandle& result, int resultSize ) override;
	void MultiplyTransposedLookupMatrixByVector( int batchSize, const CLookupMatrix& matrix,
		const CConstFloatHandle& vectorHandle, const CFloatHandle& resultHandle, int resultSize ) override;
	void MultiplyTransposedLookupMatrixByVectorAndAdd( int batchSize, const CLookupMatrix& matrix,
		const CConstFloatHandle& vectorHandle, const CFloatHandle& resultHandle, int resultSize ) override;
	void MultiplyVectorByTransposedLookupVectorAndAddToTable( int batchSize,
		const CFloatHandle& tableHandle, int vectorCount, int vectorSize, const CConstIntHandle& indices,
		const CConstFloatHandle& firstHandle, int firstSize, const CLookupVector& secondVector ) override;
	void MultiplyMatrixByTransposedMatrix( const CConstFloatHandle& firstHandle, int firstHeight,
		int firstWidth, int firstRowSize, const CConstFloatHandle& secondHandle, int secondHeight, int secondRowSize,
		const CFloatHandle& resultHandle, int resultRowSize, int resultBufferSize,
		const CSmallMatricesMultiplyDesc* desc = nullptr ) override;
	void MultiplyMatrixByTransposedMatrix( int batchSize, const CConstFloatHandle& firstHandle,
		int firstHeight, int firstWidth, const CConstFloatHandle& secondHandle, int secondHeight,
		const CFloatHandle& resultHandle, int resultBufferSize,
		const CSmallMatricesMultiplyDesc* desc = nullptr ) override;
	void MultiplySparseMatrixByTransposedMatrix( int firstHeight, int firstWidth, int secondHeight,
		const CSparseMatrixDesc& firstDesc, const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle ) override;
	void MultiplyTransposedMatrixBySparseMatrix( int firstHeight, int firstWidth, int secondWidth,
		const CConstFloatHandle& firstHandle, const CSparseMatrixDesc& secondDesc, const CFloatHandle& resultHandle,
		bool isTransposedSparse ) override;
	void MultiplyTransposedMatrixBySparseMatrixAndAdd( int firstHeight, int firstWidth, int secondWidth,
		const CConstFloatHandle& firstHandle, const CSparseMatrixDesc& secondDesc, const CFloatHandle& resultHandle ) override;
	void MultiplySparseMatrixByMatrix( int firstHeight, int firstWidth, int secondWidth,
		const CSparseMatrixDesc& firstDesc, const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle ) override;
	void MultiplyTransposedSparseMatrixByMatrix( int firstHeight, int firstWidth, int secondWidth,
		const CSparseMatrixDesc& firstDesc, const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle ) override;
	void MultiplyTransposedMatrixByMatrixAndAdd( const CConstFloatHandle& firstHandle, int firstHeight, int firstWidth, int firstRowSize,
		const CConstFloatHandle& secondHandle, int secondWidth, int secondRowSize,
		const CFloatHandle& resultHandle, int resultRowSize, int resultBufferSize,
		const CSmallMatricesMultiplyDesc* desc = nullptr ) override;
	void MultiplyTransposedMatrixByMatrix( int batchSize, const CConstFloatHandle& firstHandle, int firstHeight, int firstWidth,
		const CConstFloatHandle& secondHandle, int secondWidth, const CFloatHandle& resultHandle, int resultBufferSize,
		const CSmallMatricesMultiplyDesc* desc = nullptr ) override;
	void MultiplyDiagMatrixByMatrix( const CConstFloatHandle& firstHandle, int firstSize,
		const CConstFloatHandle& secondHandle, int secondWidth,
		const CFloatHandle& resultHandle, int resultBufferSize ) override;
	void Multiply1DiagMatrixByMatrix( int batchSize, const CConstFloatHandle& firstHandle, int firstSize,
		const CConstFloatHandle& secondHandle, int secondWidth,
		const CFloatHandle& resultHandle, int resultBufferSize ) override;
	void MultiplyMatrixByMatrix( int batchSize, const CConstFloatHandle& firstHandle, int firstHeight,
		int firstWidth, const CConstFloatHandle& secondHandle, int secondWidth,
		const CFloatHandle& resultHandle, int resultBufferSize,
		const CSmallMatricesMultiplyDesc* desc = nullptr ) override;
	void BatchMultiplyMatrixByDiagMatrix( int batchSize, const CConstFloatHandle& firstHandle, int height,
		int width, int firstMatrixOffset, const CConstFloatHandle& secondHandle, int secondMatrixOffset,
		const CFloatHandle& resultHandle, int resultBufferSize ) override;
	void TransposeMatrix( int batchSize, const CConstFloatHandle& firstHandle,
		int height, int medium, int width, int channels, const CFloatHandle& resultHandle, int resultBufferSize ) override;
	void TransposeMatrix( int batchSize, const CConstIntHandle& firstHandle,
		int height, int medium, int width, int channels, const CIntHandle& resultHandle, int resultBufferSize ) override;
	void MultiplyDiagMatrixByMatrixAndAdd( int batchSize, const CConstFloatHandle& firstHandle,
		int firstSize, const CConstFloatHandle& secondHandle, int secondWidth, const CFloatHandle& resultHandle ) override;
	void RowMultiplyMatrixByMatrix( const CConstFloatHandle& firstHandle,
		const CConstFloatHandle& secondHandle, int height, int width, const CFloatHandle& result ) override;
	void MatrixSpreadRows( const CConstFloatHandle& sourceHandle, int height, int width,
		const CFloatHandle& resultHandle, int resultHeight, const CConstIntHandle& indexHandle,
		const CConstFloatHandle& fillValue ) override;
	void MatrixSpreadRowsAdd( const CConstFloatHandle& sourceHandle, int height, int width,
		const CFloatHandle& resultHandle, int resultHeight, const CConstIntHandle& indexHandle ) override;
	void MatrixSpreadRows( const CConstIntHandle& sourceHandle, int height, int width,
		const CIntHandle& resultHandle, int resultHeight, const CConstIntHandle& indexHandle,
		const CConstIntHandle& fillValue ) override;
	void SingularValueDecomposition( const CFloatHandle& a, int height, int width, const CFloatHandle& u, const CFloatHandle& s,
		const CFloatHandle& vt, const CFloatHandle& superb, bool returnLeftVectors, bool returnRightVectors ) override;
	void QRFactorization( int height, int width, const CFloatHandle& matrixHandle, const CFloatHandle* qHandle, const CFloatHandle* rHandle,
		bool inplace, bool returnQ, bool returnR ) override;
	void LUFactorization( int height, int width, const CFloatHandle& matrixHandle ) override;

	// IDnnEngine interface methods
	void BlobMergeByDim( TBlobDim dim, const CBlobDesc* from, const CFloatHandle* fromData, int fromCount,
		const CBlobDesc& to, const CFloatHandle& toData ) override;
	void BlobMergeByDim( TBlobDim dim, const CBlobDesc* from, const CIntHandle* fromData, int fromCount,
		const CBlobDesc& to, const CIntHandle& toData ) override;
	void BlobSplitByDim( TBlobDim dim, const CBlobDesc& from, const CConstFloatHandle& fromData,
		const CBlobDesc* to, const CFloatHandle* toData, int toCount ) override;
	void BlobSplitByDim( TBlobDim dim, const CBlobDesc& from, const CConstIntHandle& fromData,
		const CBlobDesc* to, const CIntHandle* toData, int toCount ) override;
	void BlobResizeImage( const CBlobDesc& from, const CFloatHandle& fromData, int deltaLeft, int deltaRight,
		int deltaTop, int deltaBottom, TBlobResizePadding padding, float defaultValue,
		const CBlobDesc& to, const CFloatHandle& toData ) override;
	void BlobGetSubSequence( const CBlobDesc& from, const CFloatHandle& fromData, const CIntHandle& indexHandle, const CBlobDesc& to,
	const CFloatHandle& toData, int startPos, bool isRev ) override;
	CTimeConvolutionDesc* InitTimeConvolution( const CBlobDesc& source, int stride, int paddingFront, int paddingBack,
		int dilation, const CBlobDesc& filter, const CBlobDesc& result ) override;
	void BlobTimeConvolution( const CTimeConvolutionDesc& desc, const CConstFloatHandle& source,
		const CConstFloatHandle& filter, const CConstFloatHandle& freeTerm, const CFloatHandle& result ) override;
	void BlobTimeConvolutionBackward( const CTimeConvolutionDesc& desc, const CConstFloatHandle& outputDiff,
		const CConstFloatHandle& filter, const CConstFloatHandle& freeTerm, const CFloatHandle& inputDiff ) override;
	void BlobTimeConvolutionLearnAdd( const CTimeConvolutionDesc& desc, const CConstFloatHandle& input,
		const CConstFloatHandle& outputDiff, const CFloatHandle& filterDiff, const CFloatHandle& freeTermDiff ) override;
	C3dConvolutionDesc* InitBlob3dConvolution( const CBlobDesc& input,
		int paddingHeight, int paddingWidth, int paddingDepth,
		int strideHeight, int strideWidth, int strideDepth,
		const CBlobDesc& filter, const CBlobDesc& output ) override;
	void Blob3dConvolution( const C3dConvolutionDesc& desc, const CConstFloatHandle& source,
		const CConstFloatHandle& filter, const CConstFloatHandle* freeTerm, const CFloatHandle& result ) override;
	void Blob3dConvolutionBackward( const C3dConvolutionDesc& desc, const CConstFloatHandle& source,
		const CConstFloatHandle& filter, const CConstFloatHandle* freeTerm, const CFloatHandle& result ) override;
	void Blob3dConvolutionLearnAdd( const C3dConvolutionDesc& desc, const CConstFloatHandle& input,
		const CConstFloatHandle& outputDiff, const CFloatHandle& filterDiff,
		const CFloatHandle* freeTermDiff, bool isFreeTermDiffFromInput ) override;
	CConvolutionDesc* InitBlobConvolution( const CBlobDesc& input,
		int paddingHeight, int paddingWidth, int strideHeight, int strideWidth, int dilationHeight, int dilationWidth,
		const CBlobDesc& filter, const CBlobDesc& output ) override;
	void BlobConvolution( const CConvolutionDesc& desc,
		const CConstFloatHandle& source, const CConstFloatHandle& filter, const CConstFloatHandle* freeTerm,
		const CFloatHandle& result ) override;
	void BlobConvolutionBackward( const CConvolutionDesc& desc, const CConstFloatHandle& outputDiff,
		const CConstFloatHandle& filter, const CConstFloatHandle* freeTerm, const CFloatHandle& inputDiff ) override;
	void BlobConvolutionLearnAdd( const CConvolutionDesc& desc,
		const CConstFloatHandle& input, const CConstFloatHandle& outputDiff, const CFloatHandle& filterDiff,
		const CFloatHandle* freeTermDiff, bool isFreeTermDiffFromInput ) override;
	CChannelwiseConvolutionDesc* InitBlobChannelwiseConvolution( const CBlobDesc& input,
		int paddingHeight, int paddingWidth, int strideHeight, int strideWidth,
		const CBlobDesc& filter, const CBlobDesc* freeTerm, const CBlobDesc& output ) override;
	void BlobChannelwiseConvolution( const CChannelwiseConvolutionDesc& desc,
		const CConstFloatHandle& source, const CConstFloatHandle& filter, const CConstFloatHandle* freeTerm, const CFloatHandle& result ) override;
	void BlobChannelwiseConvolutionBackward( const CChannelwiseConvolutionDesc& convDesc,
		const CConstFloatHandle& source, const CConstFloatHandle& filter, const CFloatHandle& result ) override;
	void BlobChannelwiseConvolutionLearnAdd( const CChannelwiseConvolutionDesc& convDesc,
		const CConstFloatHandle& input, const CConstFloatHandle& outputDiff, const CFloatHandle& filterDiff,
		const CFloatHandle* freeTermDiff ) override;

	CSmallMatricesMultiplyDesc* InitSmallMatricesMultiplyDesc(
		int /*firstHeight*/, int /*firstWidth*/, int /*secondWidth*/, int /*secondRowSize*/, int /*resultWidth*/,
		bool /*resultAdd*/, bool /*trans1*/, bool /*trans2*/ ) const override
	{ return new CSmallMatricesMultiplyDesc{}; }

	CGlobalMaxPoolingDesc* InitGlobalMaxPooling( const CBlobDesc& source, const CBlobDesc& maxIndices, const CBlobDesc& result ) override;
	void BlobGlobalMaxPooling( const CGlobalMaxPoolingDesc& desc,
		const CConstFloatHandle& source, const CIntHandle& maxIndices, const CFloatHandle& result ) override;
	void BlobGlobalMaxPoolingBackward( const CGlobalMaxPoolingDesc& desc,
		const CConstFloatHandle& resultDiff, const CConstIntHandle& maxIndices, const CFloatHandle& sourceDiff ) override;
	C3dMaxPoolingDesc* Init3dMaxPooling( const CBlobDesc& source,
		int filterHeight, int filterWidth, int filterDepth,
		int strideHeight, int strideWidth, int strideDepth, const CBlobDesc& result ) override;
	void Blob3dMaxPooling( const C3dMaxPoolingDesc& desc, const CConstFloatHandle& source,
		const CIntHandle* maxIndices, const CFloatHandle& result ) override;
	void Blob3dMaxPoolingBackward( const C3dMaxPoolingDesc& desc, const CConstFloatHandle& resultDiff,
		const CConstIntHandle& maxIndices, const CFloatHandle& sourceDiff ) override;
	C3dMeanPoolingDesc* Init3dMeanPooling( const CBlobDesc& source,
		int filterHeight, int filterWidth, int filterDepth,
		int strideHeight, int strideWidth, int strideDepth,
		const CBlobDesc& result ) override;
	void Blob3dMeanPooling( const C3dMeanPoolingDesc& desc, const CConstFloatHandle& source, const CFloatHandle& result ) override;
	void Blob3dMeanPoolingBackward( const C3dMeanPoolingDesc& desc, const CConstFloatHandle& resultDiff, const CFloatHandle& sourceDiff ) override;
	CMaxPoolingDesc* InitMaxPooling( const CBlobDesc& source,
		int filterHeight, int filterWidth, int strideHeight, int strideWidth,
		const CBlobDesc& result ) override;
	void BlobMaxPooling( const CMaxPoolingDesc& desc, const CConstFloatHandle& source,
		const CIntHandle* maxIndices, const CFloatHandle& result ) override;
	void BlobMaxPoolingBackward( const CMaxPoolingDesc& desc, const CConstFloatHandle& resultDiff,
		const CConstIntHandle& maxIndices, const CFloatHandle& sourceDiff ) override;
	CMaxOverTimePoolingDesc* InitMaxOverTimePooling( const CBlobDesc& source, int filterLen, int strideLen, const CBlobDesc& result ) override;
	void BlobMaxOverTimePooling( const CMaxOverTimePoolingDesc& desc, const CConstFloatHandle& source,
		const CIntHandle* maxIndices, const CFloatHandle& result ) override;
	void BlobMaxOverTimePoolingBackward( const CMaxOverTimePoolingDesc& desc, const CConstFloatHandle& resultDiff,
		const CConstIntHandle& maxIndices, const CFloatHandle& sourceDiff ) override;
	CMeanPoolingDesc* InitMeanPooling( const CBlobDesc& source,
		int filterHeight, int filterWidth, int strideHeight, int strideWidth,
		const CBlobDesc& result ) override;
	void BlobMeanPooling( const CMeanPoolingDesc& desc, const CConstFloatHandle& source, const CFloatHandle& result ) override;
	void BlobMeanPoolingBackward( const CMeanPoolingDesc& desc, const CConstFloatHandle& resultDiff, const CFloatHandle& sourceDiff ) override;
	CGlobalMaxOverTimePoolingDesc* InitGlobalMaxOverTimePooling( const CBlobDesc& source, const CBlobDesc& result ) override;
	void BlobGlobalMaxOverTimePooling( const CGlobalMaxOverTimePoolingDesc& desc, const CConstFloatHandle& source,
		const CIntHandle* maxIndices, const CFloatHandle& result ) override;
	void BlobGlobalMaxOverTimePoolingBackward( const CGlobalMaxOverTimePoolingDesc& desc, const CConstFloatHandle& resultDiff,
		const CConstIntHandle& maxIndices, const CFloatHandle& sourceDiff ) override;
	void Upsampling2DForward( const CBlobDesc& input, const CConstIntHandle& inputData, int heightCopyCount,
		int widthCopyCount, const CBlobDesc& result, const CIntHandle& resultData ) override;
	void Upsampling2DForward( const CBlobDesc& input, const CConstFloatHandle& inputData, int heightCopyCount,
		int widthCopyCount, const CBlobDesc& result, const CFloatHandle& resultData ) override;
	void Upsampling2DBackward( const CBlobDesc& input, const CConstFloatHandle& inputData, int heightCopyCount,
		int widthCopyCount, const CBlobDesc& result, const CFloatHandle& resultData ) override;
	void BuildIntegerHist( const CConstIntHandle& numbersHandle, int numbersCount,
		const CIntHandle& resultHandle, int maxNumber ) override;
	void MatrixRowsToVectorSquaredL2Distance( const CConstFloatHandle& matrixHandle, const int matrixHeight,
		const int matrixWidth, const CConstFloatHandle& vectorHandle, const CFloatHandle& resultHandle ) override;
	CRleConvolutionDesc* InitBlobRleConvolution( const CBlobDesc& input, float strokeValue,
		float nonStrokeValue, int strideHeight, int strideWidth, const CBlobDesc& filter,
		const CBlobDesc& output ) override;
	void BlobRleConvolution( const CRleConvolutionDesc& desc, const CConstFloatHandle& source,
		const CConstFloatHandle& filter, const CConstFloatHandle* freeTerm, const CFloatHandle& result ) override;
	void BlobRleConvolutionLearnAdd( const CRleConvolutionDesc& desc, const CConstFloatHandle& input,
		const CConstFloatHandle& outputDiff, const CFloatHandle& filterDiff, const CFloatHandle* freeTermDiff ) override;
	void Reorg( const CBlobDesc& source, const CFloatHandle& sourceData, int stride, bool isForward,
		const CBlobDesc& result, const CFloatHandle& resultData ) override;
	void Reorg( const CBlobDesc& source, const CIntHandle& sourceData, int stride, bool isForward,
		const CBlobDesc& result, const CIntHandle& resultData ) override;
	void SpaceToDepth( const CBlobDesc& source, const CConstFloatHandle& sourceData, int blockSize,
		const CBlobDesc& result, const CFloatHandle& resultData ) override;
	void SpaceToDepth( const CBlobDesc& source, const CConstIntHandle& sourceData, int blockSize,
		const CBlobDesc& result, const CIntHandle& resultData ) override;
	void DepthToSpace( const CBlobDesc& source, const CConstFloatHandle& sourceData, int blockSize,
		const CBlobDesc& result, const CFloatHandle& resultData ) override;
	void DepthToSpace( const CBlobDesc& source, const CConstIntHandle& sourceData, int blockSize,
		const CBlobDesc& result, const CIntHandle& resultData ) override;
	void AddWidthIndex( const CBlobDesc& source, const CConstFloatHandle& sourceData, bool isForward, const CFloatHandle& result ) override;
	void AddWidthIndex( const CBlobDesc& source, const CConstIntHandle& sourceData, bool isForward, const CIntHandle& result ) override;
	void AddHeightIndex( const CBlobDesc& source, const CConstFloatHandle& sourceData, bool isForward, const CFloatHandle& result ) override;
	void AddHeightIndex( const CBlobDesc& source, const CConstIntHandle& sourceData, bool isForward, const CIntHandle& result ) override;
	CDropoutDesc* InitDropout( float rate, bool isSpatial, bool isBatchwise, const CBlobDesc& input,
		const CBlobDesc& output, int seed ) override;
	void Dropout( const CDropoutDesc& desc, const CFloatHandle& input, const CFloatHandle& output ) override;
	void QrnnFPooling( bool reverse, int sequenceLength, int objectSize,
		const CConstFloatHandle& update, const CConstFloatHandle& forget, const CConstFloatHandle& initialState,
		const CFloatHandle& result ) override;
	void QrnnFPoolingBackward( bool reverse, int sequenceLength, int objectSize,
		const CConstFloatHandle& update, const CConstFloatHandle& forget,
		const CConstFloatHandle& initialState, const CConstFloatHandle& result, const CFloatHandle& resultDiff,
		const CFloatHandle& updateDiff, const CFloatHandle& forgetDiff ) override;
	void QrnnIfPooling( bool reverse, int sequenceLength, int objectSize,
		const CConstFloatHandle& update, const CConstFloatHandle& forget, const CConstFloatHandle& input,
		const CConstFloatHandle& initialState, const CFloatHandle& result ) override;
	void QrnnIfPoolingBackward( bool reverse, int sequenceLength, int objectSize,
		const CConstFloatHandle& update, const CConstFloatHandle& forget, const CConstFloatHandle& input,
		const CConstFloatHandle& initialState, const CConstFloatHandle& result, const CFloatHandle& resultDiff,
		const CFloatHandle& updateDiff, const CFloatHandle& forgetDiff, const CFloatHandle& inputDiff ) override;
	void IndRnnRecurrent( bool reverse, int sequenceLength, int batchSize, int objectSize, TActivationFunction activation,
		const CConstFloatHandle& wx, const CConstFloatHandle& mask, const CConstFloatHandle& u,
		const CFloatHandle& h ) override;
	void IndRnnRecurrentBackward( bool reverse, int sequenceLength, int batchSize, int objectSize, TActivationFunction activation,
		const CConstFloatHandle& mask, const CConstFloatHandle& u, const CConstFloatHandle& h, const CConstFloatHandle& hDiff,
		const CFloatHandle& wxDiff ) override;
	void IndRnnRecurrentLearn( bool reverse, int sequenceLength, int batchSize, int objectSize, TActivationFunction activation,
		const CConstFloatHandle& mask, const CConstFloatHandle& u, const CConstFloatHandle& h, const CConstFloatHandle& hDiff,
		const CFloatHandle& uDiff ) override;
	CLrnDesc* InitLrn( const CBlobDesc& source, int windowSize, float bias, float alpha, float beta ) override;
	void Lrn( const CLrnDesc& desc, const CConstFloatHandle& input, const CFloatHandle& invSum,
		const CFloatHandle& invSumBeta, const CFloatHandle& outputHandle ) override;
	void LrnBackward( const CLrnDesc& desc, const CConstFloatHandle& input, const CConstFloatHandle& output,
		const CConstFloatHandle& outputDiff, const CConstFloatHandle& invSum, const CConstFloatHandle& invSumBeta,
		const CFloatHandle& inputDiff ) override;
	void CtcLossForward( int resultLen, int batchSize, int classCount, int labelLen, int blankLabel, bool skipBlanks,
		const CConstFloatHandle& result, const CConstIntHandle& labels,
		const CConstIntHandle& labelLens, const CConstIntHandle& resultLens, const CConstFloatHandle& labelWeights,
		const CFloatHandle& loss, const CFloatHandle& lossGradient ) override;
	void BertConv( const CConstFloatHandle& dataHandle, const CConstFloatHandle& kernelHandle, int seqLen, int batchSize,
		int numHeads, int headSize, int kernelSize, const CFloatHandle& outputHandle ) override;
	void BertConvBackward( const CConstFloatHandle& dataHandle, const CConstFloatHandle& kernelHandle,
		const CConstFloatHandle& outDiffHandle, int seqLen, int batchSize, int numHeads, int headSize, int kernelSize,
		const CFloatHandle& dataDiffHandle, const CFloatHandle& kernelDiffHandle ) override;
	CLstmDesc* InitLstm( int hiddenSize, int objectSize,
		const CConstFloatHandle& inputWeights, const CConstFloatHandle& inputFreeTerm,
		const CConstFloatHandle& recurrentWeights, const CConstFloatHandle& recurrentFreeTerm ) override;
	void Lstm( CLstmDesc& desc, bool reverse, int sequenceLength, int sequenceCount,
		const CConstFloatHandle& inputStateBackLink, const CConstFloatHandle& inputMainBackLink,
		const CConstFloatHandle& input, const CFloatHandle& outputStateBackLink,
		const CFloatHandle& outputMainBackLink ) override;
	void LinearInterpolation( const CConstFloatHandle& dataHandle, const CFloatHandle& resultHandle,
		TInterpolationCoords coords, TInterpolationRound round, int objectCount, int scaledAxis,
		int objectSize, float scale ) override;
	void ScatterND( const CConstIntHandle& indicesHandle, const CConstFloatHandle& updatesHandle,
		const CFloatHandle& dataHandle, const CBlobDesc& dataDesc, int updateCount, int indexDims ) override;
	void ScatterND( const CConstIntHandle& indicesHandle, const CConstIntHandle& updatesHandle,
		const CIntHandle& dataHandle, const CBlobDesc& dataDesc, int updateCount, int indexDims ) override;
	void ChannelwiseWith1x1( const CBlobDesc& inputDesc, const CBlobDesc& outputDesc,
		const CRowwiseOperationDesc& rowwiseDesc, const CChannelwiseConvolutionDesc& convDesc,
		const CConstFloatHandle& inputHandle, const CFloatHandle& outputHandle ) override;
	void MobileNetV2Block( const CBlobDesc& inputDesc, const CBlobDesc& outputDesc,
		const CRowwiseOperationDesc& rowwiseDesc, const CChannelwiseConvolutionDesc& convDesc,
		const CConstFloatHandle& inputHandle, const CFloatHandle& outputHandle ) override;
	void MobileNetV3PreSEBlock( const CBlobDesc& inputDesc, const CBlobDesc& outputDesc,
		const CChannelwiseConvolutionDesc& convDesc, const CConstFloatHandle& inputHandle,
		const CConstFloatHandle& expandFilter, const CConstFloatHandle* expandFreeTerm,
		TActivationFunction expandActivation, float expandReluParam, const CConstFloatHandle& channelwiseFilter,
		const CConstFloatHandle* channelwiseFreeTerm, TActivationFunction channelwiseActivation,
		float channelwiseReluParam, const CFloatHandle& outputHandle, const CSmallMatricesMultiplyDescsArray* descs = nullptr ) override;
	void MobileNetV3PostSEBlock( const CBlobDesc& channelwiseOutputDesc, int outputChannels,
		const CConstFloatHandle& channelwiseOutputHandle, const CConstFloatHandle& squeezeAndExciteHandle,
		const CConstFloatHandle* residualHandle, TActivationFunction activation, float reluParam,
		const CConstFloatHandle& downFilterHandle, const CConstFloatHandle* downFreeTermHandle,
		const CFloatHandle& outputHandle, const CSmallMatricesMultiplyDescsArray* descs = nullptr ) override;
	CSmallMatricesMultiplyDescsArray* InitSmallMatricesMultiplyDescsArray() override { return new CSmallMatricesMultiplyDescsArray{}; }
	// WARNING: Rowwise computation is ineffective on GPUs
	CRowwiseOperationDesc* InitRowwiseActivation( const CActivationDesc& desc ) override;
	CRowwiseOperationDesc* InitRowwiseChWith1x1( int stride, const CConstFloatHandle& channelwiseFilter,
		const CConstFloatHandle* channelwiseFreeTerm, TActivationFunction activation, float reluParam,
		const CConstFloatHandle& convFilter, const CConstFloatHandle* convFreeTerm,
		int outputChannels, bool residual ) override;
	CRowwiseOperationDesc* InitRowwiseConv( int paddingHeight, int paddingWidth, int strideHeight,
		int strideWidth, int dilationHeight, int dilationWidth, const CBlobDesc& filterDesc,
		const CConstFloatHandle& filter, const CConstFloatHandle* freeTerm ) override;
	CRowwiseOperationDesc* InitRowwiseChConv( int paddingHeight, int paddingWidth, int strideHeight,
		int strideWidth, const CBlobDesc& filterDesc, const CConstFloatHandle& filter,
		const CConstFloatHandle* freeTerm ) override;
	CRowwiseOperationDesc* InitRowwiseResizeImage( TBlobResizePadding padding, float defaultValue,
		int deltaLeft, int deltaRight, int deltaTop, int deltaBottom ) override;
	CRowwiseOperationDesc* InitRowwiseMobileNetV2( int inputChannels,
		const CConstFloatHandle& expandFilter, const CConstFloatHandle* expandFreeTerm, int expandedChannels,
		TActivationFunction expandActivation, float expandReluParam,
		const CConstFloatHandle& channelwiseFilter, const CConstFloatHandle* channelwiseFreeTerm, int stride,
		TActivationFunction channelwiseActivation, float channelwiseReluParam,
		const CConstFloatHandle& downFilter, const CConstFloatHandle* downFreeTerm,
		int outputChannels, bool residual ) override;
	CRowwiseOperationDesc* InitRowwise2DPooling( bool isMax, int filterHeight, int filterWidth,
		int strideHeight, int strideWidth ) override;
	CBlobDesc RowwiseReshape( CRowwiseOperationDesc** operations, int operationCount,
		const CBlobDesc& input ) override;
	void RowwiseExecute( const CBlobDesc& inputDesc, CRowwiseOperationDesc** operations, int operationCount,
		const CFloatHandle& input, const CFloatHandle& output ) override;

	IPerformanceCounters* CreatePerformanceCounters() const override { 	return new CPerformanceCountersDefault(); }
	void AllReduce( const CFloatHandle& handle, int size ) override;
	void Broadcast( const CFloatHandle& handle, int size, int root ) override;
	void AbortDistributed() override;
	CMathEngineDistributedInfo GetDistributedInfo() override { return distributedInfo; }
	bool IsDistributed() override { return distributedInfo.Threads > 1; }
#ifdef NEOML_USE_NCCL
	void SetDistributedCommunicator( const ncclUniqueId& uniqueId, const CMathEngineDistributedInfo& info,
		std::shared_ptr<std::atomic<bool>> isAbort );
#endif
protected:
	// IRawMemoryManager interface methods
	CMemoryHandle Alloc( size_t size ) override;
	void Free( const CMemoryHandle& handle ) override;

private:
	CDllLoader loader; // loader to guarantee the correctness of dlls' loads/frees
	const CCusparse* cusparse; // cusparse library functions
	const CCublas* cublas; // cublas library functions

	const float* cudaConstZero; // pointer to __constant__ == 0.f
	const float* cudaConstOne; // pointer to __constant__ == 1.f

	mutable std::mutex mutex; // protects the data below
	std::unique_ptr<CCudaDevice> device; // the device descriptor
	cublasHandle_t cublasHandle; // cublas library handle
	cusparseHandle_t cusparseHandle; // cusparse library handle
	std::unique_ptr<CMemoryPool> memoryPool; // memory manager
	std::unique_ptr<CDeviceStackAllocator> deviceStackRunTime; // GPU memory stack allocator
	std::unique_ptr<CHostStackAllocator> hostStackRunTime; // regular memory stack allocator
	CMathEngineDistributedInfo distributedInfo;
#ifdef NEOML_USE_NCCL
	std::unique_ptr<CCudaDistributedCommunicator> ncclCommunicator = nullptr;
#endif

	IMathEngine& mathEngine() { IMathEngine* engine = this; return *engine; }
	CCudaDevice* captureCudaDevice( int deviceNumber, size_t memoryLimit );
	CCudaDevice* captureSpecifiedCudaDevice( int deviceNumber, size_t memoryLimit, bool reuseDevice );

	int alignXSizeForWarp( int xSize );
	int getCudaTempMatrixMaxHeight( int matrixHeight, int matrixWidth );
	void getCudaTaskGrid( int& blockCount, int& threadCount, int taskCount, int combineCount = 1 );
	void getCudaTaskGrid2D( dim3& blockCount, dim3& threadCount, int height, int width, int _maxThreadCount = UINT_MAX );
	void getCudaTaskGrid3D( dim3& blockCount, dim3& threadCount, int batchSize, int height, int width, int _maxThreadCount = UINT_MAX );
	void getCudaTaskGrid2DMinYX( int minY, int minX, dim3& blockCount, dim3& threadCount, int height, int width, int _maxThreadCount = UINT_MAX );
	void getCudaTaskGrid3DMinZYX( int minZ, int minY, int minX, dim3& blockCount, dim3& threadCount,
		int batchSize, int height, int width, int _maxThreadCount = UINT_MAX );

	template<class T>
	void transposeMatrixImpl( int batchSize, const CTypedMemoryHandle<const T>& firstHandle,
		int height, int medium, int width, int channels, const CTypedMemoryHandle<T>& resultHandle, int resultBufferSize );
	void batchAddVectorToMatrixColumns( bool isNeg, int batchSize, const CConstFloatHandle& matrixHandle, const CFloatHandle& resultHandle,
		int matrixHeight, int matrixWidth, const CConstFloatHandle& vectorHandle );
	void sumMatrixColumnsKernelFunc( const CFloatHandle& resultHandle, const float* matrix,
		int matrixHeight, int matrixWidth, bool isNeg );
	void multiplyVectorByLookupMatrixImpl( int batchSize, const CLookupMatrix& matrix,
		const CConstFloatHandle& vectorHandle, const CFloatHandle& resultHandle, int resultSize, bool isAdd );
	template<class T>
	void matrixSpreadRowsImpl( const T* source, int height, int width,
		CTypedMemoryHandle<T> result, int resultHeight, const int* index, const CTypedMemoryHandle<const T>& fillValue );
	template<class TInput, class TLookup>
	void vectorMultichannelLookupAndCopy( int batchSize, int channelCount, const CTypedMemoryHandle<const TInput>& inputHandle,
		const CTypedMemoryHandle<const TLookup>* lookupHandles, const CLookupDimension* lookupDimensions, int lookupCount,
		const CTypedMemoryHandle<TLookup>& outputHandle, int outputChannelsCount );
	template<class T>
	void vectorMultichannelLookupAndAddToTable( int batchSize, int channelCount, const CTypedMemoryHandle<const T>& inputHandle,
		const CFloatHandle* lookupHandles, const CLookupDimension* lookupDimensions, int lookupCount,
		const CConstFloatHandle& multHandle, const CConstFloatHandle& matrixHandle, int outputChannelsCount );

	template<class T>
	void blobMergeByDimCuda( int dimNum, const CBlobDesc* from, const CTypedMemoryHandle<T>* fromData, int fromCount,
		const CBlobDesc& to, const CTypedMemoryHandle<T>& toData );
	template<class T>
	void blobMergeByDim0( const CBlobDesc* from, const CTypedMemoryHandle<T>* fromData, int fromCount, const CTypedMemoryHandle<T>& toData );
	template<class T>
	void blobMergeByDim( int dim, const CBlobDesc* from, const CTypedMemoryHandle<T>* fromData, int fromCount, const CBlobDesc& to, const CTypedMemoryHandle<T>& toData );
	template<class T>
	void blobSplitByDimCuda( int dimNum, const CBlobDesc& from, const CTypedMemoryHandle<const T>& fromData, const CBlobDesc* to, const CTypedMemoryHandle<T>* toData, int toCount );
	template<class T>
	void blobSplitByDim0( const CBlobDesc& from, const CTypedMemoryHandle<const T>& fromData, const CBlobDesc* to, const CTypedMemoryHandle<T>* toData, int toCount );
	template<class T>
	void blobSplitByDim( int dim, const CBlobDesc& from, const CTypedMemoryHandle<const T>& fromData, const CBlobDesc* to, const CTypedMemoryHandle<T>* toData, int toCount );

	void blobConvertFromRle( const CCudaRleConvolutionDesc& desc, const CConstFloatHandle& source, const CFloatHandle& result );

	void multiplyMatrixByTransposedMatrixAndAdd( const CConstFloatHandle& firstHandle,
		int firstHeight, int firstWidth, int firstRowSize,
		const CConstFloatHandle& secondHandle, int secondHeight, int secondRowSize,
		const CFloatHandle& resultHandle, int resultRowSize );
	void multiplyMatrixByDiagMatrix( int batchSize, const CConstFloatHandle& firstHandle, int height,
		int width, int firstMatrixOffset, const CConstFloatHandle& secondHandle, int secondMatrixOffset,
		const CFloatHandle& resultHandle );

	void setVectorToMatrixElements( const CFloatHandle& matrix, int height, int width,
		const CConstIntHandle& rowIndices, const CConstIntHandle& columnIndices,
		const CConstFloatHandle& vector, int vectorSize );
	void ctcFillPadding( int maxSeqLen, int batchSize, int classCount,
		const CFloatHandle& dataHandle, const CConstIntHandle& seqLensHandle );
	void ctcCalcForwardVariables( int resultLen, int batchSize, int classCount, int padLabelLen, bool skipBlanks,
		const CConstFloatHandle& blankSkipMask, const CConstFloatHandle& resultLogProbMask, const CFloatHandle& logAlpha );
	void ctcCalcBackwardVariables( int resultLen, int batchSize, int classCount, int padLabelLen, bool skipBlanks,
		const CConstFloatHandle& blankSkipMask, const CConstFloatHandle& resultLogProbMask, const CConstIntHandle& resultLens,
		const CConstIntHandle& labelLens, const CFloatHandle& logBeta );
	void ctcCalcGradient( int resultLen, int batchSize, int classCount, int padLabelLen, bool skipBlanks,
		const CConstFloatHandle& resultProb, const CConstFloatHandle& logAlphaBeta,
		const CConstIntHandle& padLabels, const CConstIntHandle& resultLens,
		const CFloatHandle& totalLogProb, const CFloatHandle& lossGradient );
};

inline void CCudaMathEngine::VectorReLUDiffOp( const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& upperThresholdHandle )
{
	VectorReLUDiff( firstHandle, secondHandle, resultHandle, vectorSize, upperThresholdHandle );
}

inline void CCudaMathEngine::VectorLeakyReLUDiffOp( const CConstFloatHandle& firstHandle,
	const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle,
	int vectorSize, const CConstFloatHandle& alpha )
{
	VectorLeakyReLUDiff( firstHandle, secondHandle, resultHandle, vectorSize, alpha );
}

inline void CCudaMathEngine::VectorHardTanhDiffOp( const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize )
{
	VectorHardTanhDiff( firstHandle, secondHandle, resultHandle, vectorSize );
}

inline void CCudaMathEngine::SumMatrixRows( int batchSize,
	const CFloatHandle& resultHandle, const CConstFloatHandle& matrixHandle, int matrixHeight, int matrixWidth )
{
	VectorFill( resultHandle, 0.f, batchSize * matrixWidth );
	SumMatrixRowsAdd( batchSize, resultHandle, matrixHandle, matrixHeight, matrixWidth );
}

} // namespace NeoML

#endif // NEOML_USE_CUDA
