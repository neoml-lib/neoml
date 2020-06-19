﻿/* Copyright © 2017-2020 ABBYY Production LLC

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
#include <RawMemoryManager.h>
#include <mutex>
#include <memory>

namespace NeoML {

struct CCpuConvolutionDesc;
struct CCommonMaxPoolingDesc;
struct CCommon3dConvolutionDesc;
struct CCommonChannelwiseConvolutionDesc;
class CDeviceStackAllocator;
class CMemoryPool;

// Math engine that uses a CPU for calculations
class CCpuMathEngine : public IMathEngine, public IRawMemoryManager {
public:
	CCpuMathEngine( int threadCount, size_t memoryLimit );
	virtual ~CCpuMathEngine();

	// IMathEngine interface methods
	TMathEngineType GetType() const override { return MET_Cpu; }
	void SetReuseMemoryMode( bool enabled ) override;
	CMemoryHandle HeapAlloc( size_t count ) override;
	void HeapFree( const CMemoryHandle& handle ) override;
	CMemoryHandle StackAlloc( size_t count ) override;
	void StackFree( const CMemoryHandle& handle ) override;
	size_t GetFreeMemorySize() const override;
	size_t GetPeakMemoryUsage() const override;
	void CleanUp() override;
	void* GetBuffer( const CMemoryHandle& handle, size_t pos, size_t size ) override;
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
	void VectorFillBernoulli( const CFloatHandle& result, float p, int vectorSize, float value, int seed ) override;
	void FilterSmallValues( const CFloatHandle& data, int dataSize, float threshold ) override;
	void VectorCopy(const CFloatHandle& first, const CConstFloatHandle& second, int vectorSize) override;
	void VectorCopy(const CIntHandle& first, const CConstIntHandle& second, int vectorSize) override;
	void VectorSum(const CConstFloatHandle& firstHandle, int vectorSize, const CFloatHandle& resultHandle) override;
	void VectorSumAdd(const CConstFloatHandle& firstHandle, int vectorSize, const CFloatHandle& resultHandle) override;
	void VectorNegSum(const CConstFloatHandle& firstHandle, int vectorSize, const CFloatHandle& resultHandle) override;
	virtual void VectorEqual( const CConstIntHandle& firstHandle, const CConstIntHandle& secondHandle,
		const CFloatHandle& resultHandle, int vectorSize ) override;
	virtual void VectorEqualValue( const CConstIntHandle& firstHandle,
		const CFloatHandle& resultHandle, int vectorSize, const CConstIntHandle& valueHandle ) override;
	virtual void VectorELU(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle,
		int vectorSize, const CConstFloatHandle& alpha) override;
	virtual void VectorELUDiff(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
		const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& alpha) override;
	virtual void VectorELUDiffOp(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
		const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& alpha) override;
	virtual void VectorReLU(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize,
		const CConstFloatHandle& upperThresholdHandle) override;
	virtual void VectorReLUDiff(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
		const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& upperThresholdHandle) override;
	virtual void VectorReLUDiffOp(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
		const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& upperThresholdHandle) override;
	virtual void VectorLeakyReLU( const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle,
		int vectorSize, const CConstFloatHandle& alpha ) override;
	virtual void VectorLeakyReLUDiff( const CConstFloatHandle& firstHandle,
		const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle,
		int vectorSize, const CConstFloatHandle& alpha ) override;
	virtual void VectorLeakyReLUDiffOp( const CConstFloatHandle& firstHandle,
		const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle,
		int vectorSize, const CConstFloatHandle& alpha ) override;
	virtual void VectorHSwish( const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, 
		int vectorSize ) override;
	virtual void VectorHSwishDiff( const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle, 
		const CFloatHandle& resultHandle, int vectorSize ) override;
	virtual void VectorEltwiseMax(const CConstFloatHandle& firstHandle,
		const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle, int vectorSize) override;
	virtual void VectorEltwiseMin(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
	const CFloatHandle& resultHandle, int vectorSize) override;
	void VectorAbs(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize) override;
	virtual void VectorAbsDiff(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
		const CFloatHandle& resultHandle, int vectorSize) override;
	void VectorHinge(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize) override;
	virtual void VectorHingeDiff(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
		const CFloatHandle& resultHandle, int vectorSize) override;
	void VectorSquaredHinge(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize) override;
	virtual void VectorSquaredHingeDiff(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
		const CFloatHandle& resultHandle, int vectorSize) override;
	void VectorHuber(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize) override;
	void VectorHuberDerivative(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize) override;
	void VectorHardTanh(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize) override;
	virtual void VectorHardTanhDiff(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
		const CFloatHandle& resultHandle, int vectorSize) override;
	virtual void VectorHardTanhDiffOp(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
		const CFloatHandle& resultHandle, int vectorSize) override;
	virtual void VectorHardSigmoid(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize, 
		const CConstFloatHandle& slopeHandle, const CConstFloatHandle& biasHandle ) override;
	virtual void VectorHardSigmoidDiff(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
		const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& slopeHandle, const CConstFloatHandle& biasHandle ) override;
	virtual void VectorHardSigmoidDiffOp(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
		const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& slopeHandle, const CConstFloatHandle& biasHandle ) override;
	void VectorExp(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize) override;
	virtual void VectorLog( const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle,
		int vectorSize ) override;
	void VectorNegLog(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize) override;
	virtual void VectorBernulliKLDerivative(const CConstFloatHandle& estimationHandle,
		const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& target) override;
	virtual void VectorAdd(const CConstFloatHandle& firstHandle,
		const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle, int vectorSize) override;
	virtual void VectorAdd( const CConstIntHandle& firstHandle,
		const CConstIntHandle& secondHandle, const CIntHandle& resultHandle, int vectorSize ) override;
	virtual void VectorAddValue(const CConstFloatHandle& firstHandle,
		const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& addition) override;
	virtual void VectorAddValue( const CConstIntHandle& firstHandle,
		const CIntHandle& resultHandle, int vectorSize, const CConstIntHandle& addition ) override;
	virtual void VectorSub(const CConstFloatHandle& firstHandle,
		const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle, int vectorSize) override;
	virtual void VectorMultiplyAndAdd(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
		const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& multHandle) override;
	virtual void VectorMultiplyAndSub(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
		const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& multHandle) override;
	virtual void VectorMultiply(const CConstFloatHandle& firstHandle,
		const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& multiplierHandle) override;
	virtual void VectorNegMultiply(const CConstFloatHandle& firstHandle,
		const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& multiplierHandle) override;
	virtual void VectorEltwiseMultiply(const CConstFloatHandle& firstHandle,
		const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle, int vectorSize) override;
	virtual void VectorEltwiseMultiplyAdd(const CConstFloatHandle& firstHandle,
		const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle, int vectorSize) override;
	virtual void VectorEltwiseNegMultiply(const CConstFloatHandle& firstHandle,
		const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle, int vectorSize) override;
	virtual void VectorEltwiseDivide(const CConstFloatHandle& firstHandle,
		const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle, int vectorSize) override;
	virtual void VectorEltwisePower(const CConstFloatHandle& firstHandle,
		const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle, int vectorSize) override;
	void VectorSqrt(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize) override;
	void VectorInv(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize) override;
	virtual void VectorMinMax(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize,
		const CConstFloatHandle& minHandle, const CConstFloatHandle& maxHandle) override;
	void VectorSigmoid(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize) override;
	virtual void VectorSigmoidDiff(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
		const CFloatHandle& resultHandle, int vectorSize) override;
	virtual void VectorSigmoidDiffOp(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
		const CFloatHandle& resultHandle, int vectorSize) override;
	void VectorTanh(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize) override;
	virtual void VectorTanhDiff(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
		const CFloatHandle& resultHandle, int vectorSize) override;
	virtual void VectorTanhDiffOp(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
		const CFloatHandle& resultHandle, int vectorSize) override;
	void VectorPower(float exponent, const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize) override;
	virtual void VectorPowerDiff(float exponent, const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
		const CFloatHandle& resultHandle, int vectorSize) override;
	virtual void VectorPowerDiffOp(float exponent, const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
		const CFloatHandle& resultHandle, int vectorSize) override;
	virtual void VectorL1DiffAdd(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
		const CFloatHandle& resultHandle, int vectorSize,
		const CConstFloatHandle& hubertThresholdHandle, const CConstFloatHandle& multHandle) override;
	virtual void VectorDotProduct(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle, int vectorSize,
		const CFloatHandle& resultHandle) override;
	void VectorEltwiseNotNegative( const CConstIntHandle& firstHanle, const CFloatHandle& resultHandle, int vectorSize ) override;
	virtual void VectorFindMaxValueInSet( const CConstFloatHandle* vectors, int vectorCount, const CFloatHandle& resultHandle,
		int vectorSize ) override;
	virtual void VectorFindMaxValueInSet( const CConstFloatHandle* vectors, int vectorCount, const CFloatHandle& resultHandle,
		const CIntHandle& indexHandle, int vectorSize ) override;
	virtual void VectorSpreadValues( const CConstFloatHandle& sourceHandle, CFloatHandle* vectors, int vectorCount,
		const CConstIntHandle& indexHandle, int vectorSize ) override;
	virtual void VectorEltwiseLogSumExp( const CConstFloatHandle& first, const CConstFloatHandle& second,
		const CFloatHandle& result, int vectorSize ) override;

	// IBlasEngine interface methods
	virtual void SetVectorToMatrixRows(const CFloatHandle& resultHandle, int matrixHeight,
		int matrixWidth, const CConstFloatHandle& vectorHandle) override;
	virtual void AddVectorToMatrixElements(const CFloatHandle& matrix, int height, int width,
		const CConstIntHandle& indices, const CConstFloatHandle& vector) override;
	virtual void AddVectorToMatrixElements(const CFloatHandle& matrix, int height, int width,
		const CConstIntHandle& rowIndices, const CConstIntHandle& columnIndices,
		const CConstFloatHandle& vector, int vectorSize) override;
	virtual void SetVectorToMatrixElements( const CFloatHandle& matrix, int height, int width,
		const CConstIntHandle& rowIndices, const CConstIntHandle& columnIndices, const CConstFloatHandle& vector, int vectorSize ) override;
	virtual void EltwiseLogSumExpVectorToMatrixElements(const CFloatHandle& matrix, int height, int width,
		const CConstIntHandle& indices, const CConstFloatHandle& vector) override;
	virtual void EltwiseLogSumExpVectorToMatrixElements(const CFloatHandle& matrix, int height, int width,
		const CConstIntHandle& rowIndices, const CConstIntHandle& columnIndices,
		const CConstFloatHandle& vector, int vectorSize) override;
	virtual void AddMatrixElementsToVector(const CConstFloatHandle& matrix, int height, int width,
		const CConstIntHandle& indices, const CFloatHandle& result, int vectorSize) override;
	virtual void AddMatrixElementsToVector(const CConstFloatHandle& matrix, int height, int width,
		const CConstIntHandle& rowIndices, const CConstIntHandle& columnIndices,
		const CFloatHandle& result, int vectorSize) override;
	virtual void AddMatrixElementsToMatrix(const CConstFloatHandle& matrix, int height, int width,
		const CFloatHandle& result, const CConstIntHandle& indices) override;
	virtual void AddVectorToMatrixRows(int batchSize, const CConstFloatHandle& matrixHandle, const CFloatHandle& resultHandle,
		int matrixHeight, int matrixWidth, const CConstFloatHandle& vectorHandle) override;
	virtual void AddVectorToMatrixColumns(const CConstFloatHandle& matrixHandle, const CFloatHandle& resultHandle,
		int matrixHeight, int matrixWidth, const CConstFloatHandle& vectorHandle) override;
	virtual void AddVectorToMatrixColumns( const CConstIntHandle& matrixHandle, const CIntHandle& resultHandle,
		int matrixHeight, int matrixWidth, const CConstIntHandle& vectorHandle ) override;
	virtual void SubVectorFromMatrixColumns(const CConstFloatHandle& matrixHandle, const CFloatHandle& resultHandle,
		int matrixHeight, int matrixWidth, const CConstFloatHandle& vectorHandle) override;
	virtual void SumMatrixRowsAdd(int batchSize, const CFloatHandle& resultHandle, const CConstFloatHandle& matrixHandle,
		int matrixHeight, int matrixWidth) override;
	virtual void SumMatrixRows(int batchSize, const CFloatHandle& resultHandle, const CConstFloatHandle& matrixHandle,
		int matrixHeight, int matrixWidth) override;
	virtual void SumMatrixColumns(const CFloatHandle& resultHandle, const CConstFloatHandle& matrixHandle,
		int matrixHeight, int matrixWidth) override;
	void MatrixLogSumExpByRows(const CConstFloatHandle& matrix, int height, int width, const CFloatHandle& result, int resultSize) override;
	void MatrixSoftmaxByRows(const CConstFloatHandle& matrix, int height, int width, const CFloatHandle& result) override;
	virtual void MatrixSoftmaxDiffOpByRows(const CConstFloatHandle& first, const CConstFloatHandle& second,
		int height, int width, const CFloatHandle& result) override;
	void MatrixLogSumExpByColumns(const CConstFloatHandle& matrix, int height, int width, const CFloatHandle& result, int resultSize) override;
	virtual void MatrixSoftmaxByColumns(const CConstFloatHandle& matrix, int height, int width,
		const CFloatHandle& result) override;
	virtual void MatrixSoftmaxDiffOpByColumns(const CConstFloatHandle& first, const CConstFloatHandle& second,
		int height, int width, const CFloatHandle& result) override;
	virtual void FindMaxValueInRows(const CConstFloatHandle& matrixHandle, int matrixHeight, int matrixWidth,
		const CFloatHandle& resultHandle, const CIntHandle& columnIndices, int vectorSize) override;
	virtual void FindMaxValueInRows(const CConstFloatHandle& matrixHandle,
		int matrixHeight, int matrixWidth, const CFloatHandle& resultHandle, int vectorSize) override;
	virtual void FindMaxValueInColumns( int batchSize, const CConstFloatHandle& matrixHandle, int matrixHeight,
		int matrixWidth, const CFloatHandle& resultHandle, const CIntHandle& rowIndices, int vectorSize ) override;
	virtual void FindMinValueInColumns( const CConstFloatHandle& matrixHandle, int matrixHeight, int matrixWidth,
		const CFloatHandle& resultHandle, const CIntHandle& columnIndices ) override;
	virtual void VectorMultichannelLookupAndCopy(int batchSize, int channelCount, const CConstFloatHandle& inputHandle,
		const CConstFloatHandle* lookupHandles, const CLookupDimension* lookupDimensions, int lookupCount, 
		const CFloatHandle& outputHandle, int outputChannels) override;
	virtual void VectorMultichannelLookupAndCopy(int batchSize, int channelCount, const CConstIntHandle& inputHandle,
		const CConstFloatHandle* lookupHandles, const CLookupDimension* lookupDimensions, int lookupCount,
		const CFloatHandle& outputHandle, int outputChannels) override;
	virtual void VectorMultichannelLookupAndAddToTable(int batchSize, int channelCount, const CConstFloatHandle& inputHandle,
		const CFloatHandle* lookupHandles, const CLookupDimension* lookupDimensions, int lookupCount, 
		const CConstFloatHandle& multHandle, const CConstFloatHandle& matrixHandle, int outputChannels) override;
	virtual void VectorMultichannelLookupAndAddToTable(int batchSize, int channelCount, const CConstIntHandle& inputHandle,
		const CFloatHandle* lookupHandles, const CLookupDimension* lookupDimensions, int lookupCount,
		const CConstFloatHandle& multHandle, const CConstFloatHandle& matrixHandle, int outputChannels) override;
	virtual void LookupAndSum( const CConstIntHandle& indicesHandle, int batchSize, int indexCount,
		const CConstFloatHandle& tableHandle, int vectorSize, const CFloatHandle& result ) override;
	virtual void LookupAndAddToTable( const CConstIntHandle& indicesHandle, int batchSize, int indexCount,
		const CConstFloatHandle& additionsHandle, int vectorSize, const CFloatHandle& tableHandle, int vectorCount ) override;
	virtual void EnumBinarization(int batchSize, const CConstFloatHandle& inputHandle, int enumSize,
		const CFloatHandle& resultHandle) override;
	virtual void EnumBinarization(int batchSize, const CConstIntHandle& inputHandle, int enumSize,
		const CFloatHandle& resultHandle) override;
	virtual void BitSetBinarization(int batchSize, int bitSetSize,
		const CConstIntHandle& inputHandle, int outputVectorSize, const CFloatHandle& resultHandle) override;
	virtual void MultiplyLookupMatrixByLookupVector(int batchSize, const CLookupMatrix& matrix,
		const CLookupVector& vector, const CFloatHandle& result, int resultSize) override;
	virtual void MultiplyTransposedLookupMatrixByVector(int batchSize, const CLookupMatrix& matrix,
		const CConstFloatHandle& vectorHandle, const CFloatHandle& resultHandle, int resultSize) override;
	virtual void MultiplyTransposedLookupMatrixByVectorAndAdd(int batchSize, const CLookupMatrix& matrix,
		const CConstFloatHandle& vectorHandle, const CFloatHandle& resultHandle, int resultSize) override;
	virtual void MultiplyVectorByTransposedLookupVectorAndAddToTable(int batchSize,
		const CFloatHandle& tableHandle, int vectorCount, int vectorSize, const CConstIntHandle& indices,
		const CConstFloatHandle& firstHandle, int firstSize, const CLookupVector& secondVector) override;
	virtual void MultiplyMatrixByTransposedMatrix(const CConstFloatHandle& firstHandle, int firstHeight,
		int firstWidth, int firstRowSize, const CConstFloatHandle& secondHandle, int secondHeight, int secondRowSize,
		const CFloatHandle& resultHandle, int resultRowSize, int resultBufferSize) override;
	virtual void MultiplySparseMatrixByTransposedMatrix( int firstHeight, int firstWidth, int secondHeight,
		const CSparseMatrixDesc& firstDesc, const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle ) override;
	virtual void MultiplyTransposedMatrixBySparseMatrixAndAdd( int firstHeight, int firstWidth, int secondWidth,
		const CConstFloatHandle& firstHandle, const CSparseMatrixDesc& secondDesc, const CFloatHandle& resultHandle ) override;
	virtual void MultiplyTransposedMatrixByMatrixAndAdd(const CConstFloatHandle& firstHandle, int firstHeight, int firstWidth,
		int firstRowSize, const CConstFloatHandle& secondHandle, int secondWidth, int secondRowSize,
		const CFloatHandle& resultHandle, int resultRowSize, int resultBufferSize) override;
	virtual void MultiplyDiagMatrixByMatrix(const CConstFloatHandle& firstHandle, int firstSize,
		const CConstFloatHandle& secondHandle, int secondWidth,
		const CFloatHandle& resultHandle, int resultBufferSize) override;
	virtual void Multiply1DiagMatrixByMatrix(int batchSize, const CConstFloatHandle& firstHandle, int firstSize,
		const CConstFloatHandle& secondHandle, int secondWidth,
		const CFloatHandle& resultHandle, int resultBufferSize) override;
	virtual void MultiplyMatrixByMatrix( int batchSize, const CConstFloatHandle& firstHandle, int firstHeight,
		int firstWidth, const CConstFloatHandle& secondHandle, int secondWidth,
		const CFloatHandle& resultHandle, int resultBufferSize ) override;
	virtual void MultiplyMatrixByDiagMatrix(const CConstFloatHandle& firstHandle, int firstHeight, int firstWidth,
		const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle, int resultBufferSize) override;
	virtual void TransposeMatrix(int batchSize, const CConstFloatHandle& firstHandle,
		int height, int medium, int width, int channels, const CFloatHandle& resultHandle, int resultBufferSize) override;
	virtual void TransposeMatrix(int batchSize, const CConstIntHandle& firstHandle,
		int height, int medium, int width, int channels, const CIntHandle& resultHandle, int resultBufferSize) override;
	virtual void MultiplyDiagMatrixByMatrixAndAdd(int batchSize, const CConstFloatHandle& firstHandle,
		int firstSize, const CConstFloatHandle& secondHandle, int secondWidth, const CFloatHandle& resultHandle) override;
	virtual void RowMultiplyMatrixByMatrix( const CConstFloatHandle& firstHandle,
		const CConstFloatHandle& secondHandle, int height, int width, const CFloatHandle& result ) override;
	virtual void MatrixSpreadRows(const CConstFloatHandle& sourceHandle, int height, int width,
		const CFloatHandle& resultHandle, int resultHeight, const CConstIntHandle& indexHandle,
		const CConstFloatHandle& fillValue) override;
	virtual void MatrixSpreadRowsAdd(const CConstFloatHandle& sourceHandle, int height, int width,
		const CFloatHandle& resultHandle, int resultHeight, const CConstIntHandle& indexHandle) override;
	virtual void MatrixSpreadRows(const CConstIntHandle& sourceHandle, int height, int width,
		const CIntHandle& resultHandle, int resultHeight, const CConstIntHandle& indexHandle,
		const CConstIntHandle& fillValue) override;

	// IDnnEngine interface methods
	virtual void BlobMergeByDim(TBlobDim dim, const CBlobDesc* from, const CFloatHandle* fromData, int fromCount,
		const CBlobDesc& to, const CFloatHandle& toData) override;
	virtual void BlobMergeByDim(TBlobDim dim, const CBlobDesc* from, const CIntHandle* fromData, int fromCount,
		const CBlobDesc& to, const CIntHandle& toData) override;
	virtual void BlobSplitByDim(TBlobDim dim, const CBlobDesc& from, const CFloatHandle& fromData,
		const CBlobDesc* to, const CFloatHandle* toData, int toCount) override;
	virtual void BlobSplitByDim(TBlobDim dim, const CBlobDesc& from, const CIntHandle& fromData,
		const CBlobDesc* to, const CIntHandle* toData, int toCount) override;
	virtual void BlobResizeImage( const CBlobDesc& from, const CFloatHandle& fromData, int deltaLeft, int deltaRight,
		int deltaTop, int deltaBottom, float defaultValue, const CBlobDesc& to, const CFloatHandle& toData ) override;
	virtual void BlobGetSubSequence( const CBlobDesc& from, const CFloatHandle& fromData, const CIntHandle& indexHandle,
		const CBlobDesc& to, const CFloatHandle& toData, int startPos, bool isRev ) override;
	virtual CTimeConvolutionDesc* InitTimeConvolution( const CBlobDesc& source, int stride, int padding, int dilation,
		const CBlobDesc& filter, const CBlobDesc& result ) override;
	virtual void BlobTimeConvolution( const CTimeConvolutionDesc& desc, const CFloatHandle& source,
		const CFloatHandle& filter, const CFloatHandle& freeTerm, const CFloatHandle& result ) override;
	virtual void BlobTimeConvolutionBackward( const CTimeConvolutionDesc& desc, const CFloatHandle& outputDiff,
		const CFloatHandle& filter, const CFloatHandle& freeTerm, const CFloatHandle& inputDiff ) override;
	virtual void BlobTimeConvolutionLearnAdd( const CTimeConvolutionDesc& desc, const CFloatHandle& input,
		const CFloatHandle& outputDiff, const CFloatHandle& filterDiff, const CFloatHandle& freeTermDiff ) override;
	virtual C3dConvolutionDesc* InitBlob3dConvolution( const CBlobDesc& input,
		int paddingHeight, int paddingWidth, int paddingDepth, int strideHeight, int strideWidth, int strideDepth,
		const CBlobDesc& filter, const CBlobDesc& output ) override;
	virtual void Blob3dConvolution( const C3dConvolutionDesc& desc, const CFloatHandle& source,
		const CFloatHandle& filter, const CFloatHandle* freeTerm, const CFloatHandle& result ) override;
	virtual void Blob3dConvolutionBackward( const C3dConvolutionDesc& desc, const CFloatHandle& source,
		const CFloatHandle& filter, const CFloatHandle* freeTerm, const CFloatHandle& result ) override;
	virtual void Blob3dConvolutionLearnAdd( const C3dConvolutionDesc& desc, const CFloatHandle& input,
		const CFloatHandle& outputDiff, const CFloatHandle& filterDiff,
		const CFloatHandle* freeTermDiff, bool isFreeTermDiffFromInput ) override;
	virtual CConvolutionDesc* InitBlobConvolution( const CBlobDesc& input, int paddingHeight, int paddingWidth,
		int strideHeight, int strideWidth, int dilationHeight, int dilationWidth, const CBlobDesc& filter, const CBlobDesc& output ) override;
	virtual void BlobConvolution( const CConvolutionDesc& desc,
		const CFloatHandle& source, const CFloatHandle& filter, const CFloatHandle* freeTerm,
		const CFloatHandle& result ) override;
	virtual void BlobConvolutionBackward( const CConvolutionDesc& desc, const CFloatHandle& outputDiff,
		const CFloatHandle& filter, const CFloatHandle* freeTerm, const CFloatHandle& inputDiff ) override;
	virtual void BlobConvolutionLearnAdd( const CConvolutionDesc& desc,
	 const CFloatHandle& input, const CFloatHandle& outputDiff, const CFloatHandle& filterDiff,
		const CFloatHandle* freeTermDiff, bool isFreeTermDiffFromInput ) override;
	virtual CChannelwiseConvolutionDesc* InitBlobChannelwiseConvolution( const CBlobDesc& input,
		int paddingHeight, int paddingWidth, int strideHeight, int strideWidth,
		const CBlobDesc& filter, const CBlobDesc* freeTerm, const CBlobDesc& output ) override;
	virtual void BlobChannelwiseConvolution( const CChannelwiseConvolutionDesc& desc, const CConstFloatHandle& source,
		const CConstFloatHandle& filter, const CConstFloatHandle* freeTerm, const CFloatHandle& result ) override;
	virtual void BlobChannelwiseConvolutionBackward( const CChannelwiseConvolutionDesc& convDesc,
		const CFloatHandle& source, const CFloatHandle& filter, const CFloatHandle& result ) override;
	virtual void BlobChannelwiseConvolutionLearnAdd( const CChannelwiseConvolutionDesc& convDesc,
		const CFloatHandle& input, const CFloatHandle& outputDiff, const CFloatHandle& filterDiff,
		const CFloatHandle* freeTermDiff ) override;
	CGlobalMaxPoolingDesc* InitGlobalMaxPooling( const CBlobDesc& source, const CBlobDesc& maxIndices, const CBlobDesc& result ) override;
	virtual void BlobGlobalMaxPooling( const CGlobalMaxPoolingDesc& desc,
		const CFloatHandle& source, const CIntHandle& maxIndices, const CFloatHandle& result ) override;
	virtual void BlobGlobalMaxPoolingBackward( const CGlobalMaxPoolingDesc& desc,
		const CFloatHandle& outputDiff, const CIntHandle& maxIndices, const CFloatHandle& inputDiff ) override;
	virtual C3dMaxPoolingDesc* Init3dMaxPooling( const CBlobDesc& source,
		int filterHeight, int filterWidth, int filterDepth,
		int strideHeight, int strideWidth, int strideDepth, const CBlobDesc& result ) override;
	virtual void Blob3dMaxPooling( const C3dMaxPoolingDesc& desc, const CFloatHandle& source,
		const CIntHandle* maxIndices, const CFloatHandle& result ) override;
	virtual void Blob3dMaxPoolingBackward( const C3dMaxPoolingDesc& desc, const CFloatHandle& outputDiff,
		const CIntHandle& maxIndices, const CFloatHandle& inputDiff ) override;
	virtual C3dMeanPoolingDesc* Init3dMeanPooling( const CBlobDesc& source,
		int filterHeight, int filterWidth, int filterDepth,
		int strideHeight, int strideWidth, int strideDepth,
		const CBlobDesc& result ) override;
	void Blob3dMeanPooling( const C3dMeanPoolingDesc& desc, const CFloatHandle& source, const CFloatHandle& result ) override;
	void Blob3dMeanPoolingBackward( const C3dMeanPoolingDesc& desc, const CFloatHandle& outputDiff, const CFloatHandle& inputDiff) override;
	virtual CMaxPoolingDesc* InitMaxPooling( const CBlobDesc& source,
		int filterHeight, int filterWidth, int strideHeight, int strideWidth,
		const CBlobDesc& result ) override;
	virtual void BlobMaxPooling( const CMaxPoolingDesc& desc, const CFloatHandle& source,
		const CIntHandle* maxIndices, const CFloatHandle& result) override;
	virtual void BlobMaxPoolingBackward( const CMaxPoolingDesc& desc, const CFloatHandle& outputDiff,
		const CIntHandle& maxIndices, const CFloatHandle& inputDiff) override;
	CMaxOverTimePoolingDesc* InitMaxOverTimePooling( const CBlobDesc& source, int filterLen, int strideLen, const CBlobDesc& result ) override;
	virtual void BlobMaxOverTimePooling( const CMaxOverTimePoolingDesc& desc, const CFloatHandle& source,
		const CIntHandle* maxIndices, const CFloatHandle& result) override;
	virtual void BlobMaxOverTimePoolingBackward( const CMaxOverTimePoolingDesc& desc, const CFloatHandle& outputDiff,
		const CIntHandle& maxIndices, const CFloatHandle& inputDiff ) override;
	virtual CMeanPoolingDesc* InitMeanPooling( const CBlobDesc& source,
		int filterHeight, int filterWidth, int strideHeight, int strideWidth,
		const CBlobDesc& result ) override;
	void BlobMeanPooling( const CMeanPoolingDesc& desc, const CFloatHandle& source, const CFloatHandle& result ) override;
	void BlobMeanPoolingBackward( const CMeanPoolingDesc& desc, const CFloatHandle& outputDiff, const CFloatHandle& inputDiff) override;
	CGlobalMaxOverTimePoolingDesc* InitGlobalMaxOverTimePooling( const CBlobDesc& source, const CBlobDesc& result ) override;
	virtual void BlobGlobalMaxOverTimePooling( const CGlobalMaxOverTimePoolingDesc& desc, const CFloatHandle& source, const CIntHandle* maxIndices,
		const CFloatHandle& result ) override;
	virtual void BlobGlobalMaxOverTimePoolingBackward( const CGlobalMaxOverTimePoolingDesc& desc, const CFloatHandle& source, const CIntHandle& maxIndices,
		const CFloatHandle& result ) override;
	virtual void Upsampling2DForward( const CBlobDesc& input, const CFloatHandle& inputData, int heightCopyCount,
		int widthCopyCount, const CBlobDesc& result, const CFloatHandle& resultData ) override;
	virtual void Upsampling2DBackward( const CBlobDesc& input, const CFloatHandle& inputData, int heightCopyCount,
		int widthCopyCount, const CBlobDesc& result, const CFloatHandle& resultData ) override;
	virtual void BuildIntegerHist( const CConstIntHandle& numbersHandle, int numbersCount,
		const CIntHandle& resultHandle, int maxNumber ) override;
	virtual void MatrixRowsToVectorSquaredL2Distance( const CConstFloatHandle& matrixHandle, const int matrixHeight,
		const int matrixWidth, const CConstFloatHandle& vectorHandle, const CFloatHandle& resultHandle ) override;
	virtual CRleConvolutionDesc* InitBlobRleConvolution( const CBlobDesc& input, float strokeValue,
		float nonStrokeValue, int strideHeight, int strideWidth, const CBlobDesc& filter,
		const CBlobDesc& output ) override;
	virtual void BlobRleConvolution( const CRleConvolutionDesc& desc, const CFloatHandle& source,
		const CFloatHandle& filter, const CFloatHandle* freeTerm, const CFloatHandle& result ) override;
	virtual void BlobRleConvolutionLearnAdd( const CRleConvolutionDesc& desc, const CFloatHandle& input,
		const CFloatHandle& outputDiff, const CFloatHandle& filterDiff, const CFloatHandle* freeTermDiff ) override;
	virtual void Reorg( const CBlobDesc& source, const CFloatHandle& sourceData, int stride, bool isForward,
		const CBlobDesc& result, const CFloatHandle& resultData ) override;
	virtual void Reorg( const CBlobDesc& source, const CIntHandle& sourceData, int stride, bool isForward,
		const CBlobDesc& result, const CIntHandle& resultData ) override;
	void AddWidthIndex( const CBlobDesc& source, const CFloatHandle& sourceData, bool isForward, const CFloatHandle& result ) override;
	void AddWidthIndex( const CBlobDesc& source, const CIntHandle& sourceData, bool isForward, const CIntHandle& result ) override;
	void AddHeightIndex( const CBlobDesc& source, const CFloatHandle& sourceData, bool isForward, const CFloatHandle& result ) override;
	void AddHeightIndex( const CBlobDesc& source, const CIntHandle& sourceData, bool isForward, const CIntHandle& result ) override;
	virtual CDropoutDesc* InitDropout( float rate, bool isSpatial, bool isBatchwise, const CBlobDesc& input,
		const CBlobDesc& output, int seed ) override;
	void Dropout( const CDropoutDesc& desc, const CFloatHandle& input, const CFloatHandle& output ) override;

	IPerformanceCounters* CreatePerformanceCounters() const override;

protected:
	// IRawMemoryManager interface methods
	CMemoryHandle Alloc( size_t size ) override;
	void Free( const CMemoryHandle& handle ) override;

private:
	const int threadCount; // the number of threads for OMP
	const int floatAlignment; // float alignment
	const int memoryAlignment; // allocation alignment
	const std::unique_ptr<CMemoryPool> memoryPool; // the memory manager
	const std::unique_ptr<CDeviceStackAllocator> stackAllocator; // the stack memory allocator
	mutable std::mutex mutex; // to protect the allocations

	IMathEngine& mathEngine() { IMathEngine* engine = this; return *engine; }

	void blob3dConvolution1x1x1( const CBlobDesc& source, const CBlobDesc& filter, const CBlobDesc& result,
		int strideHeight, int strideWidth, int strideDepth,
		const float* sourceData, const float* filterData, const float* freeTermData, float* resultData );
	void blob3dConvolution1x1x1Backward( const CCommon3dConvolutionDesc& desc, const CFloatHandle& outputDiffData,
		const CFloatHandle& filterData, const CFloatHandle* freeTermData, const CFloatHandle& inputDiffData );
	void blob3dConvolution1x1x1LearnAdd( const CCommon3dConvolutionDesc& desc, const CFloatHandle& inputData,
		const CFloatHandle& outputDiffData, const CFloatHandle& filterDiffData, const CFloatHandle* freeTermDiffData );
	void blob3dConvolution( const CCommon3dConvolutionDesc& desc, const CFloatHandle& sourceData,
		const CFloatHandle& filterData, const CFloatHandle* freeTermData, const CFloatHandle& resultData );
	void blob3dConvolutionBackward( const CCommon3dConvolutionDesc& desc, const CFloatHandle& sourceData,
		const CFloatHandle& filterData, const CFloatHandle* freeTermData, const CFloatHandle& resultData );
	void blob3dConvolutionLearnAdd( const CCommon3dConvolutionDesc& desc,
		const CFloatHandle& inputData, const CFloatHandle& outputDiffData, const CFloatHandle& filterDiffData,
		const CFloatHandle* freeTermDiffData, bool isFreeTermDiffFromInput );
	void blob3dConvolutionPrepareInput( const CCommon3dConvolutionDesc& desc, const CFloatHandle& inputPreparedData,
		const CFloatHandle& inputBlobData, int inputObject, int outputHeight, int outputWidthExStart, int outputWidthExCount );

	void vectorCopy(const CFloatHandle& firstHandle, const CConstFloatHandle& secondHandle, int vectorSize);
	void vectorFill( const CFloatHandle& result, float value, int vectorSize );
	void setVectorToMatrixRows( const CFloatHandle& resultHandle, int matrixHeight, int matrixWidth, const CConstFloatHandle& vectorHandle );
	void addVectorToMatrixRows( const CConstFloatHandle& matrixHandle, const CFloatHandle& resultHandle,
		int matrixHeight, int matrixWidth, int matrixRowSize, int resultRowSize, const CConstFloatHandle& vectorHandle );
	void addMatrixToMatrix( const CFloatHandle& firstHandle, int height,
		int width, int firstRowSize, const CConstFloatHandle& secondHandle, int secondRowSize );
	void sumMatrixRowsAdd( const CFloatHandle& resultHandle, const CConstFloatHandle& matrixHandle,
		int matrixHeight, int matrixWidth );
	void sumMatrixColumnsAdd(const CFloatHandle& resultHandle, const CConstFloatHandle& matrixHandle,
		int matrixHeight, int matrixWidth);
	void multiplyMatrixByMatrix( const CConstFloatHandle& firstHandle, int firstHeight,
		int firstWidth, int firstRowSize, const CConstFloatHandle& secondHandle, int secondWidth, int secondRowSize,
		const CFloatHandle& resultHandle, int resultRowSize, int resultBufferSize );
	void multiplyMatrixByMatrixAndAdd( const CConstFloatHandle& firstHandle, int firstHeight,
		int firstWidth, int firstRowSize, const CConstFloatHandle& secondHandle, int secondWidth, int secondRowSize,
		const CFloatHandle& resultHandle, int resultRowSize, int resultBufferSize );
	void multiplyTransposedMatrixByMatrix( const CConstFloatHandle& firstHandle, int firstHeight, int firstWidth,
		const CConstFloatHandle& secondHandle, int secondWidth, const CFloatHandle& resultHandle, int resultBufferSize );
	void batchMultiplyTransposedMatrixByMatrix( int batchSize, const CConstFloatHandle& firstHandle, int firstHeight, int firstWidth, 
		const CConstFloatHandle& secondHandle, int secondWidth, const CFloatHandle& resultHandle, int resultBufferSize );
	void multiplyTransposedMatrixByMatrixAndAdd( const CConstFloatHandle& firstHandle, int firstHeight, int firstWidth, int firstRowSize,
		const CConstFloatHandle& secondHandle, int secondWidth, int secondRowSize,
		const CFloatHandle& resultHandle, int resultRowSize, int resultBufferSize );
	void multiplyMatrixByTransposedMatrix( const CConstFloatHandle& firstHandle, int firstHeight,
		int firstWidth, int firstRowSize, const CConstFloatHandle& secondHandle, int secondHeight, int secondRowSize,
		const CFloatHandle& resultHandle, int resultRowSize, int resultBufferSize );
	void batchMultiplyMatrixByTransposedMatrix( int batchSize, const CConstFloatHandle& firstHandle, int firstHeight,
		int firstWidth, const CConstFloatHandle& secondHandle, int secondHeight, const CFloatHandle& resultHandle, int resultBufferSize );
	void multiplyMatrixByTransposedMatrixAndAdd( const float* first, int firstHeight, int firstWidth, int firstRowSize,
		const float* second, int secondHeight, int secondRowSize, float* result, int resultRowSize );

	template<class T>
	void blobMergeByDimCommon( int dimNum, const CBlobDesc* from, const CTypedMemoryHandle<T>* fromData, int fromCount,
		const CBlobDesc& to, const CTypedMemoryHandle<T>& toData );
	template<class T>
	void blobMergeByDim0( const CBlobDesc* from, const CTypedMemoryHandle<T>* fromData, int fromCount, const CTypedMemoryHandle<T>& toData );
	template<class T>
	void blobMergeByDim( int dim, const CBlobDesc* from, const CTypedMemoryHandle<T>* fromData, int fromCount, const CBlobDesc& to, const CTypedMemoryHandle<T>& toData );
	template<class T>
	void blobSplitByDimCommon( int dimNum, const CBlobDesc& from, const CTypedMemoryHandle<T>& fromData, const CBlobDesc* to, const CTypedMemoryHandle<T>* toData, int toCount );
	template<class T>
	void blobSplitByDim0( const CTypedMemoryHandle<T>& fromData, const CBlobDesc* to, const CTypedMemoryHandle<T>* toData, int toCount );
	template<class T>
	void blobSplitByDim( int dim, const CBlobDesc& from, const CTypedMemoryHandle<T>& fromData, const CBlobDesc* to, const CTypedMemoryHandle<T>* toData, int toCount );

	template<class T>
	void transposeMatrixImpl( int batchSize, const CTypedMemoryHandle<const T>& firstHandle,
		int height, int medium, int width, int channels, const CTypedMemoryHandle<T>& resultHandle, int resultBufferSize );
	void createDilationTemporaryBlob( const CCpuConvolutionDesc& desc, const CFloatHandle& inputBlob, int inputBatch,
		int outputColumnStart, int outputColumnCount, const CFloatHandle& temporaryBlob );
	template<class TConvolutionDesc>
	void createTemporaryBlob( const TConvolutionDesc& desc, const CFloatHandle& inputData,
		int inputBatch, int outputRowStart, int outputRowCount, const CFloatHandle& tempBlob );
	void transposeResult( const CCpuConvolutionDesc& desc, const CConstFloatHandle& outputTransposedData,
		int batch, int resultStart, int resultCount, const CFloatHandle& result );
	void fillTempData( const CFloatHandle& sourceData, const CFloatHandle& filterData, const CCpuConvolutionDesc& desc, int start, int count );
	void blobConvolutionForwardAlgo0( const CCpuConvolutionDesc& desc,
		const CFloatHandle& sourceData, const CFloatHandle& filterData, const CFloatHandle* freeTerm,
		const CFloatHandle& resultData );
	void blobConvolutionForwardAlgo1( const CCpuConvolutionDesc& desc,
		const CFloatHandle& sourceData, const CFloatHandle& filterData, const CFloatHandle* freeTerm,
		const CFloatHandle& resultData );
	void backwardConvolutionAddFilterToOutput( const CCpuConvolutionDesc& desc, const CFloatHandle& temp,
		const CFloatHandle* freeTerm, const CFloatHandle& output );
	void backwardDilationConvolutionAddFilterToOutput( const CCpuConvolutionDesc& desc, const CFloatHandle& temp,
		const CFloatHandle* freeTermData, const CFloatHandle& outputData );
	void blobConvolutionBackwardAlgo1( const CCpuConvolutionDesc& desc,
		const CFloatHandle& sourceData, const CFloatHandle& filterData, const CFloatHandle* freeTerm,
		const CFloatHandle& resultData );
	void fillTempBlobsForLearnAlgo2( const CCpuConvolutionDesc& desc, const CFloatHandle& outputDiff,
		const CBlobDesc& tempBlob, const CFloatHandle& tempHandle );
	void blobConvolutionBackwardAlgo2( const CCpuConvolutionDesc& desc, const CFloatHandle& outputDiffData,
		const CFloatHandle& filterData, const CFloatHandle* freeTermData, const CFloatHandle& inputDiffData );
	void blobConvolutionLearnAlgo1( const CCpuConvolutionDesc& desc,
		const CFloatHandle& input, const CFloatHandle& outputDiff, const CFloatHandle& filterDiff,
		const CFloatHandle* freeTermDiff, bool isFreeTermDiffFromInput );
	void blobConvolutionLearnAlgo2( const CCpuConvolutionDesc& desc, const CFloatHandle& input,
		const CFloatHandle& outputDiff, const CFloatHandle& filterDiff, const CFloatHandle* freeTermDiff, bool isFreeTermDiffFromInput );

	void blobChannelwiseConvolutionFilter3x3Padding1Stride1( const CCommonChannelwiseConvolutionDesc& desc, const float* source,
		const float* filter, const float* freeTerm, float* result );
	void blobChannelwiseConvolutionFilter3x3Padding1Stride2( const CCommonChannelwiseConvolutionDesc& desc, const float* source,
		const float* filter, const float* freeTerm, float* result );

	void findMaxValueInColumns(const CFloatHandle& resultHandle, const CConstFloatHandle& matrixHandle,
		int matrixHeight, int matrixWidth);
	void findMaxValueInColumns(const CFloatHandle& resultHandle, const CIntHandle& rowIndices,
		const CConstFloatHandle& matrixHandle, int matrixHeight, int matrixWidth);
	void blobMaxPoolingWithIndices(const CCommonMaxPoolingDesc& desc, const CFloatHandle& sourceData,
		const CIntHandle& maxIndicesData, const CFloatHandle& resultData);
	void blobMaxPoolingWithoutIndices(const CCommonMaxPoolingDesc& desc, const CFloatHandle& sourceData, const CFloatHandle& resultData);
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
