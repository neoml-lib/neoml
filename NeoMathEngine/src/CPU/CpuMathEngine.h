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

#pragma once

#include <NeoMathEngine/NeoMathEngine.h>
#include <NeoMathEngine/SimdMathEngine.h>
#include <RawMemoryManager.h>
#include <DllLoader.h>
#include <mutex>
#include <memory>
#include <CpuMathEngineDnnDistributed.h>

namespace NeoML {

struct CCpuConvolutionDesc;
struct CCommonMaxPoolingDesc;
struct CCommon3dConvolutionDesc;
struct CCommonChannelwiseConvolutionDesc;
class CDeviceStackAllocator;
class CMemoryPool;
class ISimdMathEngine;

// Math engine that uses a CPU for calculations
class CCpuMathEngine : public IMathEngine, public IRawMemoryManager {
public:
	CCpuMathEngine( int threadCount, size_t memoryLimit,
		std::shared_ptr<CMultiThreadDistributedCommunicator> communicator = nullptr, 
		const CMathEngineDistributedInfo& distributedInfo = CMathEngineDistributedInfo() );
	~CCpuMathEngine() override;

	// IMathEngine interface methods
	TMathEngineType GetType() const override { return MET_Cpu; }
	void SetReuseMemoryMode( bool enabled ) override;
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
	void GetMathEngineInfo( CMathEngineInfo& info ) const override;

	// IVectorMathEngine interface methods
	void VectorFill(const CFloatHandle& result, float value, int vectorSize) override;
	void VectorFill(const CIntHandle& result, int value, int vectorSize) override;
	void VectorFill(const CFloatHandle& result, int vectorSize, const CConstFloatHandle& value) override;
	void VectorFill(const CIntHandle& result, int vectorSize, const CConstIntHandle& value) override;
	void VectorConvert(const CConstFloatHandle& from, const CIntHandle& to, int vectorSize) override;
	void VectorConvert(const CConstIntHandle& from, const CFloatHandle& to, int vectorSize) override;
	void VectorFillBernoulli( const CFloatHandle& result, float p, int vectorSize, float value, int seed ) override;
	void FilterSmallValues( const CFloatHandle& data, int dataSize, float threshold ) override;
	void VectorCopy(const CFloatHandle& first, const CConstFloatHandle& second, int vectorSize) override;
	void VectorCopy(const CIntHandle& first, const CConstIntHandle& second, int vectorSize) override;
	void BroadcastCopy(const CIntHandle& toHandle, const CConstIntHandle& fromHandle,
		const CBlobDesc& toDesc, const CBlobDesc& fromDesc, int additionalWidth) override;
	void BroadcastCopy(const CFloatHandle& toHandle, const CConstFloatHandle& fromHandle,
		const CBlobDesc& toDesc, const CBlobDesc& fromDesc, int additionalWidth) override;
	void VectorSum(const CConstFloatHandle& firstHandle, int vectorSize, const CFloatHandle& resultHandle) override;
	void VectorSumAdd(const CConstFloatHandle& firstHandle, int vectorSize, const CFloatHandle& resultHandle) override;
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
	void VectorELU(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle,
		int vectorSize, const CConstFloatHandle& alpha) override;
	void VectorELUDiff(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
		const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& alpha) override;
	void VectorELUDiffOp(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
		const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& alpha) override;
	void VectorReLU(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize,
		const CConstFloatHandle& upperThresholdHandle) override;
	void VectorReLUDiff(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
		const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& upperThresholdHandle) override;
	void VectorReLUDiffOp(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
		const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& upperThresholdHandle) override;
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
	void VectorEltwiseMax(const CConstFloatHandle& firstHandle,
		const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle, int vectorSize) override;
	void VectorEltwiseMin(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize) override;
	void VectorAbs(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize) override;
	void VectorAbsDiff(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
		const CFloatHandle& resultHandle, int vectorSize) override;
	void VectorAbsDiff(const CConstFloatHandle& sourceGradHandle, int gradHeight, int gradWidth,
		const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle) override;
	void VectorHinge(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize) override;
	void VectorHingeDiff(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
		const CFloatHandle& resultHandle, int vectorSize) override;
	void VectorSquaredHinge(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize) override;
	void VectorSquaredHingeDiff(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
		const CFloatHandle& resultHandle, int vectorSize) override;
	void VectorHuber(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize) override;
	void VectorHuberDerivative(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize) override;
	void VectorHardTanh(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize) override;
	void VectorHardTanhDiff(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
		const CFloatHandle& resultHandle, int vectorSize) override;
	void VectorHardTanhDiffOp(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
		const CFloatHandle& resultHandle, int vectorSize) override;
	void VectorHardSigmoid(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize, 
		const CConstFloatHandle& slopeHandle, const CConstFloatHandle& biasHandle ) override;
	void VectorHardSigmoidDiff(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
		const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& slopeHandle, const CConstFloatHandle& biasHandle ) override;
	void VectorHardSigmoidDiffOp(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
		const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& slopeHandle, const CConstFloatHandle& biasHandle ) override;
	void VectorNeg(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize) override;
	void VectorExp(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize) override;
	void VectorLog( const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle,
		int vectorSize ) override;
	void VectorLogDiff( const CConstFloatHandle& sourceGradHandle, int sourceGradHeight, int sourceGradWidth,
		const CConstFloatHandle& valueHandle, const CFloatHandle& resultHandle ) override;
	void VectorNegLog(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize) override;
	void VectorErf( const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize ) override;
	void VectorBernulliKLDerivative(const CConstFloatHandle& estimationHandle,
		const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& target) override;
	void VectorAdd(const CConstFloatHandle& firstHandle,
		const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle, int vectorSize) override;
	void VectorAdd( const CConstIntHandle& firstHandle,
		const CConstIntHandle& secondHandle, const CIntHandle& resultHandle, int vectorSize ) override;
	void VectorAddValue(const CConstFloatHandle& firstHandle,
		const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& addition) override;
	void VectorAddValue( const CConstIntHandle& firstHandle,
		const CIntHandle& resultHandle, int vectorSize, const CConstIntHandle& addition ) override;
	void VectorSub(const CConstIntHandle& firstHandle,
		const CConstIntHandle& secondHandle, const CIntHandle& resultHandle, int vectorSize) override;
	void VectorSub(const CConstFloatHandle& firstHandle,
		const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle, int vectorSize) override;
	void VectorSub(const CConstFloatHandle& firstHandle,
		float second, const CFloatHandle& resultHandle, int vectorSize) override;
	void VectorSub(float first,
		const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle, int vectorSize) override;
	void VectorMultiplyAndAdd(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
		const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& multHandle) override;
	void VectorMultiplyAndSub(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
		const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& multHandle) override;
	void VectorMultiply(const CConstFloatHandle& firstHandle,
		const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& multiplierHandle) override;
	void VectorMultiply(const CConstIntHandle& firstHandle,
		const CIntHandle& resultHandle, int vectorSize, const CConstIntHandle& multiplierHandle) override;
	void VectorNegMultiply(const CConstFloatHandle& firstHandle,
		const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& multiplierHandle) override;
	void VectorEltwiseMultiply(const CConstIntHandle& firstHandle,
		const CConstIntHandle& secondHandle, const CIntHandle& resultHandle, int vectorSize) override;
	void VectorEltwiseMultiply(const CConstFloatHandle& firstHandle,
		const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle, int vectorSize) override;
	void VectorEltwiseMultiplyAdd(const CConstFloatHandle& firstHandle,
		const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle, int vectorSize) override;
	void VectorEltwiseNegMultiply(const CConstFloatHandle& firstHandle,
		const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle, int vectorSize) override;
	void VectorEltwiseDivide(const CConstIntHandle& firstHandle,
		const CConstIntHandle& secondHandle, const CIntHandle& resultHandle, int vectorSize) override;
	void VectorEltwiseDivide(const CConstFloatHandle& firstHandle,
		const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle, int vectorSize) override;
	void VectorEltwisePower(const CConstFloatHandle& firstHandle,
		const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle, int vectorSize) override;
	void VectorSqrt(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize) override;
	void VectorInv(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize) override;
	void VectorMinMax(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize,
		const CConstFloatHandle& minHandle, const CConstFloatHandle& maxHandle) override;
	void VectorMinMaxDiff(const CConstFloatHandle& sourceGradHandle, int gradHeight, int gradWidth,
		const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle,
		const CConstFloatHandle& minHandle, const CConstFloatHandle& maxHandle) override;
	void VectorSigmoid(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize) override;
	void VectorSigmoidDiff(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
		const CFloatHandle& resultHandle, int vectorSize) override;
	void VectorSigmoidDiffOp(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
		const CFloatHandle& resultHandle, int vectorSize) override;
	void VectorTanh(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize) override;
	void VectorTanhDiff(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
		const CFloatHandle& resultHandle, int vectorSize) override;
	void VectorTanhDiffOp(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
		const CFloatHandle& resultHandle, int vectorSize) override;
	void VectorPower(float exponent, const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize) override;
	void VectorPowerDiff(float exponent, const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
		const CFloatHandle& resultHandle, int vectorSize) override;
	void VectorPowerDiffOp(float exponent, const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
		const CFloatHandle& resultHandle, int vectorSize) override;
	void VectorL1DiffAdd(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
		const CFloatHandle& resultHandle, int vectorSize,
		const CConstFloatHandle& hubertThresholdHandle, const CConstFloatHandle& multHandle) override;
	void VectorDotProduct(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle, int vectorSize,
		const CFloatHandle& resultHandle) override;
	void VectorEltwiseNot( const CConstIntHandle& firstHandle, const CIntHandle& resultHandle, int vectorSize ) override;
	void VectorEltwiseNotNegative( const CConstIntHandle& firstHanle, const CFloatHandle& resultHandle, int vectorSize ) override;
	void VectorEltwiseLess( const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
		const CFloatHandle& resultHandle, int vectorSize ) override;
	void VectorEltwiseLess( const CConstFloatHandle& firstHandle, float second,
		const CFloatHandle& resultHandle, int vectorSize ) override;
	void VectorEltwiseLess( float first, const CConstFloatHandle& secondHandle,
		const CFloatHandle& resultHandle, int vectorSize ) override;
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
	void VectorFindMaxValueInSet( const CConstFloatHandle* vectors, int vectorCount, const CFloatHandle& resultHandle,
		int vectorSize ) override;
	void VectorFindMaxValueInSet( const CConstFloatHandle* vectors, int vectorCount, const CFloatHandle& resultHandle,
		const CIntHandle& indexHandle, int vectorSize ) override;
	void VectorSpreadValues( const CConstFloatHandle& sourceHandle, CFloatHandle* vectors, int vectorCount,
		const CConstIntHandle& indexHandle, int vectorSize ) override;
	void VectorTopK(const CConstFloatHandle& first, int firstSize, int k, const CFloatHandle& result, const CIntHandle& indices) override;
	void VectorTopKDiff(const CConstFloatHandle& sourceGrad, int sourceGradHeight, int sourceGradWidth,
		const CConstIntHandle& indices, int k, const CFloatHandle& resultGrad) override;

	// IBlasEngine interface methods
	void SetVectorToMatrixRows(const CFloatHandle& resultHandle, int matrixHeight,
		int matrixWidth, const CConstFloatHandle& vectorHandle) override;
	void AddVectorToMatrixElements(const CFloatHandle& matrix, int height, int width,
		const CConstIntHandle& indices, const CConstFloatHandle& vector) override;
	void AddVectorToMatrixElements(const CFloatHandle& matrix, int height, int width,
		const CConstIntHandle& rowIndices, const CConstIntHandle& columnIndices,
		const CConstFloatHandle& vector, int vectorSize) override;
	void AddMatrixElementsToVector(const CConstFloatHandle& matrix, int height, int width,
		const CConstIntHandle& indices, const CFloatHandle& result, int vectorSize) override;
	void AddMatrixElementsToVector(const CConstFloatHandle& matrix, int height, int width,
		const CConstIntHandle& rowIndices, const CConstIntHandle& columnIndices,
		const CFloatHandle& result, int vectorSize) override;
	void AddDiagMatrixToMatrix( const CConstFloatHandle& diagMatrix, const CConstFloatHandle& matrix,
		int height, int width, const CFloatHandle& result ) override;
	void AddMatrixElementsToMatrix(const CConstFloatHandle& matrix, int height, int width,
		const CFloatHandle& result, const CConstIntHandle& indices) override;
	void AddVectorToMatrixRows(int batchSize, const CConstFloatHandle& matrixHandle, const CFloatHandle& resultHandle,
		int matrixHeight, int matrixWidth, const CConstFloatHandle& vectorHandle) override;
	void AddVectorToMatrixColumns(const CConstFloatHandle& matrixHandle, const CFloatHandle& resultHandle,
		int matrixHeight, int matrixWidth, const CConstFloatHandle& vectorHandle) override;
	void AddVectorToMatrixColumns( const CConstIntHandle& matrixHandle, const CIntHandle& resultHandle,
		int matrixHeight, int matrixWidth, const CConstIntHandle& vectorHandle ) override;
	void SubVectorFromMatrixColumns(const CConstFloatHandle& matrixHandle, const CFloatHandle& resultHandle,
		int matrixHeight, int matrixWidth, const CConstFloatHandle& vectorHandle) override;
	void SumMatrixRowsAdd(int batchSize, const CFloatHandle& resultHandle, const CConstFloatHandle& matrixHandle,
		int matrixHeight, int matrixWidth) override;
	void SumMatrixRows(int batchSize, const CFloatHandle& resultHandle, const CConstFloatHandle& matrixHandle,
		int matrixHeight, int matrixWidth) override;
	void SumMatrixRows(int batchSize, const CIntHandle& resultHandle, const CConstIntHandle& matrixHandle,
		int matrixHeight, int matrixWidth) override;
	void SumMatrixColumns(const CFloatHandle& resultHandle, const CConstFloatHandle& matrixHandle,
		int matrixHeight, int matrixWidth) override;
	void MatrixColumnsEltwiseDivide( const CConstFloatHandle& matrix, int matrixHeight, int matrixWidth,
		const CConstFloatHandle& vector, const CFloatHandle& resultHandle ) override;
	void MatrixLogSumExpByRows(const CConstFloatHandle& matrix, int height, int width, const CFloatHandle& result, int resultSize) override;
	void MatrixSoftmaxByRows(const CConstFloatHandle& matrix, int height, int width, const CFloatHandle& result) override;
	void MatrixSoftmaxDiffOpByRows(const CConstFloatHandle& first, const CConstFloatHandle& second,
		int height, int width, const CFloatHandle& result) override;
	void MatrixSoftmaxByColumns(const CConstFloatHandle& matrix, int height, int width,
		const CFloatHandle& result) override;
	void MatrixSoftmaxDiffOpByColumns(const CConstFloatHandle& first, const CConstFloatHandle& second,
		int height, int width, const CFloatHandle& result) override;
	void FindMaxValueInRows(const CConstFloatHandle& matrixHandle, int matrixHeight, int matrixWidth,
		const CFloatHandle& resultHandle, const CIntHandle& columnIndices, int vectorSize) override;
	void FindMaxValueInRows(const CConstFloatHandle& matrixHandle,
		int matrixHeight, int matrixWidth, const CFloatHandle& resultHandle, int vectorSize) override;
	void FindMaxValueInColumns( int batchSize, const CConstFloatHandle& matrixHandle, int matrixHeight,
		int matrixWidth, const CFloatHandle& resultHandle, const CIntHandle& rowIndices, int vectorSize ) override;
	void FindMinValueInColumns( const CConstFloatHandle& matrixHandle, int matrixHeight, int matrixWidth,
		const CFloatHandle& resultHandle, const CIntHandle& columnIndices ) override;
	void VectorMultichannelLookupAndCopy(int batchSize, int channelCount, const CConstFloatHandle& inputHandle,
		const CConstFloatHandle* lookupHandles, const CLookupDimension* lookupDimensions, int lookupCount, 
		const CFloatHandle& outputHandle, int outputChannels) override;
	void VectorMultichannelLookupAndCopy(int batchSize, int channelCount, const CConstIntHandle& inputHandle,
		const CConstFloatHandle* lookupHandles, const CLookupDimension* lookupDimensions, int lookupCount,
		const CFloatHandle& outputHandle, int outputChannels) override;
	void VectorMultichannelLookupAndCopy(int batchSize, int channelCount, const CConstIntHandle& inputHandle,
		const CConstIntHandle* lookupHandles, const CLookupDimension* lookupDimensions, int lookupCount,
		const CIntHandle& outputHandle, int outputChannels) override;
	void VectorMultichannelLookupAndAddToTable(int batchSize, int channelCount, const CConstFloatHandle& inputHandle,
		const CFloatHandle* lookupHandles, const CLookupDimension* lookupDimensions, int lookupCount, 
		const CConstFloatHandle& multHandle, const CConstFloatHandle& matrixHandle, int outputChannels) override;
	void VectorMultichannelLookupAndAddToTable(int batchSize, int channelCount, const CConstIntHandle& inputHandle,
		const CFloatHandle* lookupHandles, const CLookupDimension* lookupDimensions, int lookupCount,
		const CConstFloatHandle& multHandle, const CConstFloatHandle& matrixHandle, int outputChannels) override;
	void LookupAndSum( const CConstIntHandle& indicesHandle, int batchSize, int indexCount,
		const CConstFloatHandle& tableHandle, int vectorSize, const CFloatHandle& result ) override;
	void LookupAndAddToTable( const CConstIntHandle& indicesHandle, int batchSize, int indexCount,
		const CConstFloatHandle& additionsHandle, int vectorSize, const CFloatHandle& tableHandle, int vectorCount ) override;
	void EnumBinarization(int batchSize, const CConstFloatHandle& inputHandle, int enumSize,
		const CFloatHandle& resultHandle) override;
	void EnumBinarization(int batchSize, const CConstIntHandle& inputHandle, int enumSize,
		const CFloatHandle& resultHandle) override;
	void BitSetBinarization(int batchSize, int bitSetSize,
		const CConstIntHandle& inputHandle, int outputVectorSize, const CFloatHandle& resultHandle) override;
	void MultiplyLookupMatrixByLookupVector(int batchSize, const CLookupMatrix& matrix,
		const CLookupVector& vector, const CFloatHandle& result, int resultSize) override;
	void MultiplyTransposedLookupMatrixByVector(int batchSize, const CLookupMatrix& matrix,
		const CConstFloatHandle& vectorHandle, const CFloatHandle& resultHandle, int resultSize) override;
	void MultiplyTransposedLookupMatrixByVectorAndAdd(int batchSize, const CLookupMatrix& matrix,
		const CConstFloatHandle& vectorHandle, const CFloatHandle& resultHandle, int resultSize) override;
	void MultiplyVectorByTransposedLookupVectorAndAddToTable(int batchSize,
		const CFloatHandle& tableHandle, int vectorCount, int vectorSize, const CConstIntHandle& indices,
		const CConstFloatHandle& firstHandle, int firstSize, const CLookupVector& secondVector) override;
	void MultiplyMatrixByTransposedMatrix(const CConstFloatHandle& firstHandle, int firstHeight,
		int firstWidth, int firstRowSize, const CConstFloatHandle& secondHandle, int secondHeight, int secondRowSize,
		const CFloatHandle& resultHandle, int resultRowSize, int resultBufferSize) override;
	void MultiplyMatrixByTransposedMatrix(int batchSize, const CConstFloatHandle& firstHandle, int firstHeight, int firstWidth,
		const CConstFloatHandle& secondHandle, int secondHeight, const CFloatHandle& resultHandle, int resultBufferSize) override;
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
	void MultiplyTransposedMatrixByMatrixAndAdd(const CConstFloatHandle& firstHandle, int firstHeight, int firstWidth,
		int firstRowSize, const CConstFloatHandle& secondHandle, int secondWidth, int secondRowSize,
		const CFloatHandle& resultHandle, int resultRowSize, int resultBufferSize) override;
	void MultiplyTransposedMatrixByMatrix(int batchSize, const CConstFloatHandle& firstHandle, int firstHeight, int firstWidth,
		const CConstFloatHandle& secondHandle, int secondWidth, const CFloatHandle& resultHandle, int resultBufferSize) override;
	void MultiplyDiagMatrixByMatrix(const CConstFloatHandle& firstHandle, int firstSize,
		const CConstFloatHandle& secondHandle, int secondWidth,
		const CFloatHandle& resultHandle, int resultBufferSize) override;
	void Multiply1DiagMatrixByMatrix(int batchSize, const CConstFloatHandle& firstHandle, int firstSize,
		const CConstFloatHandle& secondHandle, int secondWidth,
		const CFloatHandle& resultHandle, int resultBufferSize) override;
	void MultiplyMatrixByMatrix( int batchSize, const CConstFloatHandle& firstHandle, int firstHeight,
		int firstWidth, const CConstFloatHandle& secondHandle, int secondWidth,
		const CFloatHandle& resultHandle, int resultBufferSize ) override;
	void MultiplyMatrixByDiagMatrix(const CConstFloatHandle& firstHandle, int firstHeight, int firstWidth,
		const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle, int resultBufferSize) override;
	void TransposeMatrix(int batchSize, const CConstFloatHandle& firstHandle,
		int height, int medium, int width, int channels, const CFloatHandle& resultHandle, int resultBufferSize) override;
	void TransposeMatrix(int batchSize, const CConstIntHandle& firstHandle,
		int height, int medium, int width, int channels, const CIntHandle& resultHandle, int resultBufferSize) override;
	void MultiplyDiagMatrixByMatrixAndAdd(int batchSize, const CConstFloatHandle& firstHandle,
		int firstSize, const CConstFloatHandle& secondHandle, int secondWidth, const CFloatHandle& resultHandle) override;
	void RowMultiplyMatrixByMatrix( const CConstFloatHandle& firstHandle,
		const CConstFloatHandle& secondHandle, int height, int width, const CFloatHandle& result ) override;
	void MatrixSpreadRows(const CConstFloatHandle& sourceHandle, int height, int width,
		const CFloatHandle& resultHandle, int resultHeight, const CConstIntHandle& indexHandle,
		const CConstFloatHandle& fillValue) override;
	void MatrixSpreadRowsAdd(const CConstFloatHandle& sourceHandle, int height, int width,
		const CFloatHandle& resultHandle, int resultHeight, const CConstIntHandle& indexHandle) override;
	void MatrixSpreadRows(const CConstIntHandle& sourceHandle, int height, int width,
		const CIntHandle& resultHandle, int resultHeight, const CConstIntHandle& indexHandle,
		const CConstIntHandle& fillValue) override;
	void SingularValueDecomposition( const CFloatHandle& a, int height, int width, const CFloatHandle& u, const CFloatHandle& s,
		const CFloatHandle& vt, const CFloatHandle& superb, bool returnLeftVectors, bool returnRightVectors ) override;
	void QRFactorization( int height, int width, const CFloatHandle& matrixHandle, const CFloatHandle* qHandle, const CFloatHandle* rHandle,
		bool inplace, bool returnQ, bool returnR ) override;

	// IDnnEngine interface methods
	void BlobMergeByDim(TBlobDim dim, const CBlobDesc* from, const CFloatHandle* fromData, int fromCount,
		const CBlobDesc& to, const CFloatHandle& toData) override;
	void BlobMergeByDim(TBlobDim dim, const CBlobDesc* from, const CIntHandle* fromData, int fromCount,
		const CBlobDesc& to, const CIntHandle& toData) override;
	void BlobSplitByDim(TBlobDim dim, const CBlobDesc& from, const CFloatHandle& fromData,
		const CBlobDesc* to, const CFloatHandle* toData, int toCount) override;
	void BlobSplitByDim(TBlobDim dim, const CBlobDesc& from, const CIntHandle& fromData,
		const CBlobDesc* to, const CIntHandle* toData, int toCount) override;
	void BlobResizeImage( const CBlobDesc& from, const CFloatHandle& fromData, int deltaLeft, int deltaRight,
		int deltaTop, int deltaBottom, float defaultValue, const CBlobDesc& to, const CFloatHandle& toData ) override;
	void BlobGetSubSequence( const CBlobDesc& from, const CFloatHandle& fromData, const CIntHandle& indexHandle,
		const CBlobDesc& to, const CFloatHandle& toData, int startPos, bool isRev ) override;
	CTimeConvolutionDesc* InitTimeConvolution( const CBlobDesc& source, int stride, int paddingFront, int paddingBack,
		int dilation, const CBlobDesc& filter, const CBlobDesc& result ) override;
	void BlobTimeConvolution( const CTimeConvolutionDesc& desc, const CConstFloatHandle& source,
		const CConstFloatHandle& filter, const CConstFloatHandle& freeTerm, const CFloatHandle& result ) override;
	void BlobTimeConvolutionBackward( const CTimeConvolutionDesc& desc, const CConstFloatHandle& outputDiff,
		const CConstFloatHandle& filter, const CConstFloatHandle& freeTerm, const CFloatHandle& inputDiff ) override;
	void BlobTimeConvolutionLearnAdd( const CTimeConvolutionDesc& desc, const CConstFloatHandle& input,
		const CConstFloatHandle& outputDiff, const CFloatHandle& filterDiff, const CFloatHandle& freeTermDiff ) override;
	C3dConvolutionDesc* InitBlob3dConvolution( const CBlobDesc& input,
		int paddingHeight, int paddingWidth, int paddingDepth, int strideHeight, int strideWidth, int strideDepth,
		const CBlobDesc& filter, const CBlobDesc& output ) override;
	void Blob3dConvolution( const C3dConvolutionDesc& desc, const CConstFloatHandle& source,
		const CConstFloatHandle& filter, const CConstFloatHandle* freeTerm, const CFloatHandle& result ) override;
	void Blob3dConvolutionBackward( const C3dConvolutionDesc& desc, const CConstFloatHandle& source,
		const CConstFloatHandle& filter, const CConstFloatHandle* freeTerm, const CFloatHandle& result ) override;
	void Blob3dConvolutionLearnAdd( const C3dConvolutionDesc& desc, const CConstFloatHandle& input,
		const CConstFloatHandle& outputDiff, const CFloatHandle& filterDiff,
		const CFloatHandle* freeTermDiff, bool isFreeTermDiffFromInput ) override;
	CConvolutionDesc* InitBlobConvolution( const CBlobDesc& input, int paddingHeight, int paddingWidth,
		int strideHeight, int strideWidth, int dilationHeight, int dilationWidth, const CBlobDesc& filter, const CBlobDesc& output ) override;
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
	void BlobChannelwiseConvolution( const CChannelwiseConvolutionDesc& desc, const CConstFloatHandle& source,
		const CConstFloatHandle& filter, const CConstFloatHandle* freeTerm, const CFloatHandle& result ) override;
	void BlobChannelwiseConvolutionBackward( const CChannelwiseConvolutionDesc& convDesc,
		const CConstFloatHandle& source, const CConstFloatHandle& filter, const CFloatHandle& result ) override;
	void BlobChannelwiseConvolutionLearnAdd( const CChannelwiseConvolutionDesc& convDesc,
		const CConstFloatHandle& input, const CConstFloatHandle& outputDiff, const CFloatHandle& filterDiff,
		const CFloatHandle* freeTermDiff ) override;
	CGlobalMaxPoolingDesc* InitGlobalMaxPooling( const CBlobDesc& source, const CBlobDesc& maxIndices, const CBlobDesc& result ) override;
	void BlobGlobalMaxPooling( const CGlobalMaxPoolingDesc& desc,
		const CConstFloatHandle& source, const CIntHandle& maxIndices, const CFloatHandle& result ) override;
	void BlobGlobalMaxPoolingBackward( const CGlobalMaxPoolingDesc& desc,
		const CFloatHandle& outputDiff, const CIntHandle& maxIndices, const CFloatHandle& inputDiff ) override;
	C3dMaxPoolingDesc* Init3dMaxPooling( const CBlobDesc& source,
		int filterHeight, int filterWidth, int filterDepth,
		int strideHeight, int strideWidth, int strideDepth, const CBlobDesc& result ) override;
	void Blob3dMaxPooling( const C3dMaxPoolingDesc& desc, const CFloatHandle& source,
		const CIntHandle* maxIndices, const CFloatHandle& result ) override;
	void Blob3dMaxPoolingBackward( const C3dMaxPoolingDesc& desc, const CFloatHandle& outputDiff,
		const CIntHandle& maxIndices, const CFloatHandle& inputDiff ) override;
	C3dMeanPoolingDesc* Init3dMeanPooling( const CBlobDesc& source,
		int filterHeight, int filterWidth, int filterDepth,
		int strideHeight, int strideWidth, int strideDepth,
		const CBlobDesc& result ) override;
	void Blob3dMeanPooling( const C3dMeanPoolingDesc& desc, const CFloatHandle& source, const CFloatHandle& result ) override;
	void Blob3dMeanPoolingBackward( const C3dMeanPoolingDesc& desc, const CFloatHandle& outputDiff, const CFloatHandle& inputDiff) override;
	CMaxPoolingDesc* InitMaxPooling( const CBlobDesc& source,
		int filterHeight, int filterWidth, int strideHeight, int strideWidth,
		const CBlobDesc& result ) override;
	void BlobMaxPooling( const CMaxPoolingDesc& desc, const CFloatHandle& source,
		const CIntHandle* maxIndices, const CFloatHandle& result) override;
	void BlobMaxPoolingBackward( const CMaxPoolingDesc& desc, const CFloatHandle& outputDiff,
		const CIntHandle& maxIndices, const CFloatHandle& inputDiff) override;
	CMaxOverTimePoolingDesc* InitMaxOverTimePooling( const CBlobDesc& source, int filterLen, int strideLen, const CBlobDesc& result ) override;
	void BlobMaxOverTimePooling( const CMaxOverTimePoolingDesc& desc, const CFloatHandle& source,
		const CIntHandle* maxIndices, const CFloatHandle& result) override;
	void BlobMaxOverTimePoolingBackward( const CMaxOverTimePoolingDesc& desc, const CFloatHandle& outputDiff,
		const CIntHandle& maxIndices, const CFloatHandle& inputDiff ) override;
	CMeanPoolingDesc* InitMeanPooling( const CBlobDesc& source,
		int filterHeight, int filterWidth, int strideHeight, int strideWidth,
		const CBlobDesc& result ) override;
	void BlobMeanPooling( const CMeanPoolingDesc& desc, const CFloatHandle& source, const CFloatHandle& result ) override;
	void BlobMeanPoolingBackward( const CMeanPoolingDesc& desc, const CFloatHandle& outputDiff, const CFloatHandle& inputDiff) override;
	CGlobalMaxOverTimePoolingDesc* InitGlobalMaxOverTimePooling( const CBlobDesc& source, const CBlobDesc& result ) override;
	void BlobGlobalMaxOverTimePooling( const CGlobalMaxOverTimePoolingDesc& desc, const CFloatHandle& source, const CIntHandle* maxIndices,
		const CFloatHandle& result ) override;
	void BlobGlobalMaxOverTimePoolingBackward( const CGlobalMaxOverTimePoolingDesc& desc, const CFloatHandle& source, const CIntHandle& maxIndices,
		const CFloatHandle& result ) override;
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
	void AddWidthIndex( const CBlobDesc& source, const CFloatHandle& sourceData, bool isForward, const CFloatHandle& result ) override;
	void AddWidthIndex( const CBlobDesc& source, const CIntHandle& sourceData, bool isForward, const CIntHandle& result ) override;
	void AddHeightIndex( const CBlobDesc& source, const CFloatHandle& sourceData, bool isForward, const CFloatHandle& result ) override;
	void AddHeightIndex( const CBlobDesc& source, const CIntHandle& sourceData, bool isForward, const CIntHandle& result ) override;
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
	CLstmDesc* InitLstm( CLstmDesc* currentDesc, const CFloatHandle& inputFullyConnectedResult, const CFloatHandle& reccurentFullyConnectedResult,
		int hiddenSize, int objectCount, int objectSize ) override;
	void Lstm( CLstmDesc& desc, 
		const CFloatHandle& inputWeights, const CConstFloatHandle& inputFreeTerm,
		const CFloatHandle& recurrentWeights, const CConstFloatHandle& recurrentFreeTerm,
		const CConstFloatHandle& inputStateBackLink, const CConstFloatHandle& inputMainBackLink, const CConstFloatHandle& input,
		const CFloatHandle& outputStateBackLink, const CFloatHandle& outputMainBackLink ) override;
	void LinearInterpolation( const CConstFloatHandle& dataHandle, const CFloatHandle& resultHandle,
		TInterpolationCoords coords, TInterpolationRound round, int objectCount, int scaledAxis,
		int objectSize, float scale ) override;
	void ScatterND( const CConstIntHandle& indicesHandle, const CConstFloatHandle& updatesHandle,
		const CFloatHandle& dataHandle, const CBlobDesc& dataDesc, int updateCount, int indexDims ) override;
	void ScatterND( const CConstIntHandle& indicesHandle, const CConstIntHandle& updatesHandle,
		const CIntHandle& dataHandle, const CBlobDesc& dataDesc, int updateCount, int indexDims ) override;

	IPerformanceCounters* CreatePerformanceCounters() const override;
	void AllReduce( const CFloatHandle& handle, int size ) override;
	void Broadcast( const CFloatHandle& handle, int size, int root ) override;
	void AbortDistributed() override;
	CMathEngineDistributedInfo GetDistributedInfo() override { return distributedInfo; }
	bool IsDistributed() override { return distributedInfo.Threads > 1; }
protected:
	// IRawMemoryManager interface methods
	CMemoryHandle Alloc( size_t size ) override;
	void Free( const CMemoryHandle& handle ) override;

private:
	const int threadCount; // the number of threads for OMP
	const int floatAlignment; // float alignment
	const int memoryAlignment; // allocation alignment
	std::shared_ptr<CMultiThreadDistributedCommunicator> communicator;
	CMathEngineDistributedInfo distributedInfo;
	const std::unique_ptr<CMemoryPool> memoryPool; // the memory manager
	const std::unique_ptr<CDeviceStackAllocator> stackAllocator; // the stack memory allocator
	mutable std::mutex mutex; // to protect the allocations

	CDllLoader dllLoader; // loading library for simd instructions
	std::unique_ptr<ISimdMathEngine> simdMathEngine; // interface for using simd instructions
	SgemmFunc customSgemmFunction; // Used when it is availabled and is faster then default sgemm

	IMathEngine& mathEngine() { IMathEngine* engine = this; return *engine; }

	void blob3dConvolution1x1x1( const CBlobDesc& source, const CBlobDesc& filter, const CBlobDesc& result,
		int strideHeight, int strideWidth, int strideDepth,
		const float* sourceData, const float* filterData, const float* freeTermData, float* resultData );
	void blob3dConvolution1x1x1Backward( const CCommon3dConvolutionDesc& desc, const float* outputDiffData,
		const float* filterData, const CConstFloatHandle* freeTermData, float* inputDiffData );
	void blob3dConvolution1x1x1LearnAdd( const CCommon3dConvolutionDesc& desc, const CConstFloatHandle& inputData,
		const CConstFloatHandle& outputDiffData, const CFloatHandle& filterDiffData, const CFloatHandle* freeTermDiffData );
	void blob3dConvolution( const CCommon3dConvolutionDesc& desc, const float* sourceData,
		const float* filterData, const CConstFloatHandle* freeTermData, float* resultData );
	void blob3dConvolutionBackward( const CCommon3dConvolutionDesc& desc, const float* sourceData,
		const CConstFloatHandle& filterData, const CConstFloatHandle* freeTermData, float* resultData );
	void blob3dConvolutionLearnAdd( const CCommon3dConvolutionDesc& desc,
		const float* inputData, const float* outputDiffData, const CFloatHandle& filterDiffData,
		const CFloatHandle* freeTermDiffData, bool isFreeTermDiffFromInput );
	void blob3dConvolutionPrepareInput( const CCommon3dConvolutionDesc& desc, float* inputPreparedData,
		const float* inputBlobData, int inputObject, int outputHeight, int outputWidthExStart, int outputWidthExCount );

	void setVectorToMatrixRows( float* result, int matrixHeight, int matrixWidth, const float* vector );
	void addVectorToMatrixRows( const float* matrix, float* result,
		int matrixHeight, int matrixWidth, int matrixRowSize, int resultRowSize, const float* vector );
	void addMatrixToMatrix( float* first, int height,
		int width, int firstRowSize, const float* second, int secondRowSize );
	void sumMatrixRowsAdd( float* result, const float* matrix,
		int matrixHeight, int matrixWidth );
	void sumMatrixColumnsAdd(const CFloatHandle& resultHandle, const CConstFloatHandle& matrixHandle,
		int matrixHeight, int matrixWidth);
	void multiplyMatrixByMatrix( const float* firstHandle, int firstHeight,
		int firstWidth, int firstRowSize, const float* secondHandle, int secondWidth, int secondRowSize,
		float* resultHandle, int resultRowSize );
	void multiplyMatrixByMatrixAndAdd( const float* first, int firstHeight,
		int firstWidth, int firstRowSize, const float* second, int secondWidth, int secondRowSize,
		float* result, int resultRowSize );
	void multiplyTransposedMatrixByMatrix( const float* first, int firstHeight, int firstWidth,
		const float* second, int secondWidth, float* result );
	void batchMultiplyTransposedMatrixByMatrix( int batchSize, const float* first, int firstHeight, int firstWidth, 
		const float* second, int secondWidth, float* result );
	void multiplyTransposedMatrixByMatrixAndAdd( const float* first, int firstHeight, int firstWidth, int firstRowSize,
		const float* second, int secondWidth, int secondRowSize, float* result, int resultRowSize );
	void multiplyMatrixByTransposedMatrix( const float* first, int firstHeight,
		int firstWidth, int firstRowSize, const float* second, int secondHeight, int secondRowSize,
		float* result, int resultRowSize );
	void batchMultiplyMatrixByTransposedMatrix( int batchSize, const CConstFloatHandle& firstHandle, int firstHeight,
		int firstWidth, const CConstFloatHandle& secondHandle, int secondHeight, const CFloatHandle& resultHandle );
	void multiplyMatrixByTransposedMatrixAndAdd( const float* first, int firstHeight, int firstWidth, int firstRowSize,
		const float* second, int secondHeight, int secondRowSize, float* result, int resultRowSize );

	template<class T>
	void blobMergeByDimCommon( int dimNum, const CBlobDesc* from, const CTypedMemoryHandle<T>* fromData, int fromCount,
		const CBlobDesc& to, const CTypedMemoryHandle<T>& toData );
	template<class T>
	void blobMergeByDim0( const CBlobDesc* from, const CTypedMemoryHandle<T>* fromData, int fromCount,
		const CBlobDesc& to, const CTypedMemoryHandle<T>& toData );
	template<class T>
	void blobMergeByDim( int dim, const CBlobDesc* from, const CTypedMemoryHandle<T>* fromData, int fromCount, const CBlobDesc& to, const CTypedMemoryHandle<T>& toData );
	template<class T>
	void blobSplitByDimCommon( int dimNum, const CBlobDesc& from, const CTypedMemoryHandle<T>& fromData, const CBlobDesc* to, const CTypedMemoryHandle<T>* toData, int toCount );
	template<class T>
	void blobSplitByDim0( const CBlobDesc& from, const CTypedMemoryHandle<T>& fromData,
		const CBlobDesc* to, const CTypedMemoryHandle<T>* toData, int toCount );
	template<class T>
	void blobSplitByDim( int dim, const CBlobDesc& from, const CTypedMemoryHandle<T>& fromData, const CBlobDesc* to, const CTypedMemoryHandle<T>* toData, int toCount );

	template<typename T>
	void transposeMatrixImpl( int batchSize, const T* firstHandle,
		int height, int medium, int width, int channels, T* resultHandle );
	void transposeMatrix(int batchSize, const float* firstHandle,
		int height, int medium, int width, int channels, float* resultHandle);
	void transposeMatrix(int batchSize, const int* firstHandle,
		int height, int medium, int width, int channels, int* resultHandle);
	void createDilationTemporaryBlob( const CCpuConvolutionDesc& desc, const float* inputBlob, int inputBatch,
		int outputColumnStart, int outputColumnCount, float* temporaryBlob );
	template<class TConvolutionDesc>
	void createTemporaryBlob( const TConvolutionDesc& desc, const float* inputData,
		int inputBatch, int outputRowStart, int outputRowCount, float* tempBlob );
	void transposeResult( const CCpuConvolutionDesc& desc, const float* outputTransposedData,
		int batch, int resultStart, int resultCount, float* result );
	void fillTempData( const float* sourceData, float* filterData, const CCpuConvolutionDesc& desc, int start, int count );
	void blobConvolutionForwardAlgo0( const CCpuConvolutionDesc& desc, const float* sourceData,
		const float* filterData, const CConstFloatHandle* freeTermData, float* resultData );
	void blobConvolutionForwardAlgo1( const CCpuConvolutionDesc& desc, const float* sourceData,
		const float* filterData, const CConstFloatHandle* freeTermData, float* resultData );
	void backwardConvolutionAddFilterToOutput( const CCpuConvolutionDesc& desc, const CConstFloatHandle& temp,
		const CConstFloatHandle* freeTerm, const CFloatHandle& output );
	void backwardDilationConvolutionAddFilterToOutput( const CCpuConvolutionDesc& desc, const CConstFloatHandle& temp,
		const CConstFloatHandle* freeTermData, const CFloatHandle& outputData );
	void blobConvolutionBackwardAlgo1( const CCpuConvolutionDesc& desc,
		const CConstFloatHandle& sourceData, const CConstFloatHandle& filterData, const CConstFloatHandle* freeTerm,
		const CFloatHandle& resultData );
	void fillTempBlobsForLearnAlgo2( const CCpuConvolutionDesc& desc, const CConstFloatHandle& outputDiff,
		const CBlobDesc& tempBlob, const CFloatHandle& tempHandle );
	void blobConvolutionBackwardAlgo2( const CCpuConvolutionDesc& desc, const CConstFloatHandle& outputDiffData,
		const CConstFloatHandle& filterData, const CConstFloatHandle* freeTermData, const CFloatHandle& inputDiffData );
	void blobConvolutionLearnAlgo1( const CCpuConvolutionDesc& desc,
		const CConstFloatHandle& input, const CConstFloatHandle& outputDiff, const CFloatHandle& filterDiff,
		const CFloatHandle* freeTermDiff, bool isFreeTermDiffFromInput );
	void blobConvolutionLearnAlgo2( const CCpuConvolutionDesc& desc, const CConstFloatHandle& input,
		const CConstFloatHandle& outputDiff, const CFloatHandle& filterDiff, const CFloatHandle* freeTermDiff,
		bool isFreeTermDiffFromInput );

	void blobChannelwiseConvolutionFilter3x3Padding1Stride1( const CCommonChannelwiseConvolutionDesc& desc, const float* source,
		const float* filter, const float* freeTerm, float* result );
	void blobChannelwiseConvolutionFilter3x3Padding1Stride2( const CCommonChannelwiseConvolutionDesc& desc, const float* source,
		const float* filter, const float* freeTerm, float* result );

	void findMaxValueInColumns( float* result, const float* matrixHandle,
		int matrixHeight, int matrixWidth);
	void findMaxValueInColumns( float* resultHandle, int* rowIndices,
		const float* matrixHandle, int matrixHeight, int matrixWidth);
	void blobMaxPoolingWithIndices(const CCommonMaxPoolingDesc& desc, const float* sourceData,
		int* maxIndicesData, float* resultData);
	void blobMaxPoolingWithoutIndices(const CCommonMaxPoolingDesc& desc, const float* sourceData, float* resultData);

	void vectorEltwiseLogSumExp( const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
		const CFloatHandle& resultHandle, int vectorSize );
	void ctcCalcForwardVariables( int resultLen, int batchSize, int classCount, int padLabelLen, bool skipBlanks,
		const CConstIntHandle& rowIndices, const CConstIntHandle& padLabels, const CConstFloatHandle& blankSkipMask,
		const CConstFloatHandle& resultLogProb, const CFloatHandle& logAlpha );
	void ctcCalcBackwardVariables( int resultLen, int batchSize, int classCount, int padLabelLen, bool skipBlanks,
		const CConstIntHandle& rowIndices, const CConstIntHandle& padLabels, const CConstFloatHandle& blankSkipMask,
		const CConstFloatHandle& resultLogProb, const CConstIntHandle& resultLens, const CConstIntHandle& labelLens,
		const CFloatHandle& logBeta );
};

inline void CCpuMathEngine::VectorReLUDiffOp(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& upperThresholdHandle)
{
	VectorReLUDiff(firstHandle, secondHandle, resultHandle, vectorSize, upperThresholdHandle);
}

inline void CCpuMathEngine::VectorLeakyReLUDiffOp( const CConstFloatHandle& firstHandle,
	const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle,
	int vectorSize, const CConstFloatHandle& alpha )
{ 
	VectorLeakyReLUDiff( firstHandle, secondHandle, resultHandle, vectorSize, alpha );
}

inline void CCpuMathEngine::VectorHardTanhDiffOp(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize)
{
	VectorHardTanhDiff(firstHandle, secondHandle, resultHandle, vectorSize);
}

} // namespace NeoML
