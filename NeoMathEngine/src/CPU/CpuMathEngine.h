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
	void VectorEqual( const CConstIntHandle& firstHandle, const CConstIntHandle& secondHandle,
		const CFloatHandle& resultHandle, int vectorSize ) override;
	void VectorEqualValue( const CConstIntHandle& firstHandle,
		const CFloatHandle& resultHandle, int vectorSize, const CConstIntHandle& valueHandle ) override;
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
	void VectorExp(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize) override;
	void VectorLog( const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle,
		int vectorSize ) override;
	void VectorNegLog(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize) override;
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
	void VectorSub(const CConstFloatHandle& firstHandle,
		const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle, int vectorSize) override;
	void VectorMultiplyAndAdd(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
		const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& multHandle) override;
	void VectorMultiplyAndSub(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
		const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& multHandle) override;
	void VectorMultiply(const CConstFloatHandle& firstHandle,
		const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& multiplierHandle) override;
	void VectorNegMultiply(const CConstFloatHandle& firstHandle,
		const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& multiplierHandle) override;
	void VectorEltwiseMultiply(const CConstFloatHandle& firstHandle,
		const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle, int vectorSize) override;
	void VectorEltwiseMultiplyAdd(const CConstFloatHandle& firstHandle,
		const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle, int vectorSize) override;
	void VectorEltwiseNegMultiply(const CConstFloatHandle& firstHandle,
		const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle, int vectorSize) override;
	void VectorEltwiseDivide(const CConstFloatHandle& firstHandle,
		const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle, int vectorSize) override;
	void VectorEltwisePower(const CConstFloatHandle& firstHandle,
		const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle, int vectorSize) override;
	void VectorSqrt(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize) override;
	void VectorInv(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize) override;
	void VectorMinMax(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize,
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
	void VectorEltwiseNotNegative( const CConstIntHandle& firstHanle, const CFloatHandle& resultHandle, int vectorSize ) override;
	void VectorFindMaxValueInSet( const CConstFloatHandle* vectors, int vectorCount, const CFloatHandle& resultHandle,
		int vectorSize ) override;
	void VectorFindMaxValueInSet( const CConstFloatHandle* vectors, int vectorCount, const CFloatHandle& resultHandle,
		const CIntHandle& indexHandle, int vectorSize ) override;
	void VectorSpreadValues( const CConstFloatHandle& sourceHandle, CFloatHandle* vectors, int vectorCount,
		const CConstIntHandle& indexHandle, int vectorSize ) override;
	void VectorEltwiseLogSumExp( const CConstFloatHandle& first, const CConstFloatHandle& second,
		const CFloatHandle& result, int vectorSize ) override;

	// IBlasEngine interface methods
	void SetVectorToMatrixRows(const CFloatHandle& resultHandle, int matrixHeight,
		int matrixWidth, const CConstFloatHandle& vectorHandle) override;
	void AddVectorToMatrixElements(const CFloatHandle& matrix, int height, int width,
		const CConstIntHandle& indices, const CConstFloatHandle& vector) override;
	void AddVectorToMatrixElements(const CFloatHandle& matrix, int height, int width,
		const CConstIntHandle& rowIndices, const CConstIntHandle& columnIndices,
		const CConstFloatHandle& vector, int vectorSize) override;
	void SetVectorToMatrixElements( const CFloatHandle& matrix, int height, int width,
		const CConstIntHandle& rowIndices, const CConstIntHandle& columnIndices, const CConstFloatHandle& vector, int vectorSize ) override;
	void EltwiseLogSumExpVectorToMatrixElements(const CFloatHandle& matrix, int height, int width,
		const CConstIntHandle& indices, const CConstFloatHandle& vector) override;
	void EltwiseLogSumExpVectorToMatrixElements(const CFloatHandle& matrix, int height, int width,
		const CConstIntHandle& rowIndices, const CConstIntHandle& columnIndices,
		const CConstFloatHandle& vector, int vectorSize) override;
	void AddMatrixElementsToVector(const CConstFloatHandle& matrix, int height, int width,
		const CConstIntHandle& indices, const CFloatHandle& result, int vectorSize) override;
	void AddMatrixElementsToVector(const CConstFloatHandle& matrix, int height, int width,
		const CConstIntHandle& rowIndices, const CConstIntHandle& columnIndices,
		const CFloatHandle& result, int vectorSize) override;
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
	void SumMatrixColumns(const CFloatHandle& resultHandle, const CConstFloatHandle& matrixHandle,
		int matrixHeight, int matrixWidth) override;
	void MatrixLogSumExpByRows(const CConstFloatHandle& matrix, int height, int width, const CFloatHandle& result, int resultSize) override;
	void MatrixSoftmaxByRows(const CConstFloatHandle& matrix, int height, int width, const CFloatHandle& result) override;
	void MatrixSoftmaxDiffOpByRows(const CConstFloatHandle& first, const CConstFloatHandle& second,
		int height, int width, const CFloatHandle& result) override;
	void MatrixLogSumExpByColumns(const CConstFloatHandle& matrix, int height, int width, const CFloatHandle& result, int resultSize) override;
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
	void MultiplyTransposedMatrixBySparseMatrixAndAdd( int firstHeight, int firstWidth, int secondWidth,
		const CConstFloatHandle& firstHandle, const CSparseMatrixDesc& secondDesc, const CFloatHandle& resultHandle ) override;
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
	CTimeConvolutionDesc* InitTimeConvolution( const CBlobDesc& source, int stride, int padding, int dilation,
		const CBlobDesc& filter, const CBlobDesc& result ) override;
	void BlobTimeConvolution( const CTimeConvolutionDesc& desc, const CFloatHandle& source,
		const CFloatHandle& filter, const CFloatHandle& freeTerm, const CFloatHandle& result ) override;
	void BlobTimeConvolutionBackward( const CTimeConvolutionDesc& desc, const CFloatHandle& outputDiff,
		const CFloatHandle& filter, const CFloatHandle& freeTerm, const CFloatHandle& inputDiff ) override;
	void BlobTimeConvolutionLearnAdd( const CTimeConvolutionDesc& desc, const CFloatHandle& input,
		const CFloatHandle& outputDiff, const CFloatHandle& filterDiff, const CFloatHandle& freeTermDiff ) override;
	C3dConvolutionDesc* InitBlob3dConvolution( const CBlobDesc& input,
		int paddingHeight, int paddingWidth, int paddingDepth, int strideHeight, int strideWidth, int strideDepth,
		const CBlobDesc& filter, const CBlobDesc& output ) override;
	void Blob3dConvolution( const C3dConvolutionDesc& desc, const CFloatHandle& source,
		const CFloatHandle& filter, const CFloatHandle* freeTerm, const CFloatHandle& result ) override;
	void Blob3dConvolutionBackward( const C3dConvolutionDesc& desc, const CFloatHandle& source,
		const CFloatHandle& filter, const CFloatHandle* freeTerm, const CFloatHandle& result ) override;
	void Blob3dConvolutionLearnAdd( const C3dConvolutionDesc& desc, const CFloatHandle& input,
		const CFloatHandle& outputDiff, const CFloatHandle& filterDiff,
		const CFloatHandle* freeTermDiff, bool isFreeTermDiffFromInput ) override;
	CConvolutionDesc* InitBlobConvolution( const CBlobDesc& input, int paddingHeight, int paddingWidth,
		int strideHeight, int strideWidth, int dilationHeight, int dilationWidth, const CBlobDesc& filter, const CBlobDesc& output ) override;
	void BlobConvolution( const CConvolutionDesc& desc,
		const CFloatHandle& source, const CFloatHandle& filter, const CFloatHandle* freeTerm,
		const CFloatHandle& result ) override;
	void BlobConvolutionBackward( const CConvolutionDesc& desc, const CFloatHandle& outputDiff,
		const CFloatHandle& filter, const CFloatHandle* freeTerm, const CFloatHandle& inputDiff ) override;
	void BlobConvolutionLearnAdd( const CConvolutionDesc& desc,
	 const CFloatHandle& input, const CFloatHandle& outputDiff, const CFloatHandle& filterDiff,
		const CFloatHandle* freeTermDiff, bool isFreeTermDiffFromInput ) override;
	CChannelwiseConvolutionDesc* InitBlobChannelwiseConvolution( const CBlobDesc& input,
		int paddingHeight, int paddingWidth, int strideHeight, int strideWidth,
		const CBlobDesc& filter, const CBlobDesc* freeTerm, const CBlobDesc& output ) override;
	void BlobChannelwiseConvolution( const CChannelwiseConvolutionDesc& desc, const CConstFloatHandle& source,
		const CConstFloatHandle& filter, const CConstFloatHandle* freeTerm, const CFloatHandle& result ) override;
	void BlobChannelwiseConvolutionBackward( const CChannelwiseConvolutionDesc& convDesc,
		const CFloatHandle& source, const CFloatHandle& filter, const CFloatHandle& result ) override;
	void BlobChannelwiseConvolutionLearnAdd( const CChannelwiseConvolutionDesc& convDesc,
		const CFloatHandle& input, const CFloatHandle& outputDiff, const CFloatHandle& filterDiff,
		const CFloatHandle* freeTermDiff ) override;
	CGlobalMaxPoolingDesc* InitGlobalMaxPooling( const CBlobDesc& source, const CBlobDesc& maxIndices, const CBlobDesc& result ) override;
	void BlobGlobalMaxPooling( const CGlobalMaxPoolingDesc& desc,
		const CFloatHandle& source, const CIntHandle& maxIndices, const CFloatHandle& result ) override;
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
	void Upsampling2DForward( const CBlobDesc& input, const CFloatHandle& inputData, int heightCopyCount,
		int widthCopyCount, const CBlobDesc& result, const CFloatHandle& resultData ) override;
	void Upsampling2DBackward( const CBlobDesc& input, const CFloatHandle& inputData, int heightCopyCount,
		int widthCopyCount, const CBlobDesc& result, const CFloatHandle& resultData ) override;
	void BuildIntegerHist( const CConstIntHandle& numbersHandle, int numbersCount,
		const CIntHandle& resultHandle, int maxNumber ) override;
	void MatrixRowsToVectorSquaredL2Distance( const CConstFloatHandle& matrixHandle, const int matrixHeight,
		const int matrixWidth, const CConstFloatHandle& vectorHandle, const CFloatHandle& resultHandle ) override;
	CRleConvolutionDesc* InitBlobRleConvolution( const CBlobDesc& input, float strokeValue,
		float nonStrokeValue, int strideHeight, int strideWidth, const CBlobDesc& filter,
		const CBlobDesc& output ) override;
	void BlobRleConvolution( const CRleConvolutionDesc& desc, const CFloatHandle& source,
		const CFloatHandle& filter, const CFloatHandle* freeTerm, const CFloatHandle& result ) override;
	void BlobRleConvolutionLearnAdd( const CRleConvolutionDesc& desc, const CFloatHandle& input,
		const CFloatHandle& outputDiff, const CFloatHandle& filterDiff, const CFloatHandle* freeTermDiff ) override;
	void Reorg( const CBlobDesc& source, const CFloatHandle& sourceData, int stride, bool isForward,
		const CBlobDesc& result, const CFloatHandle& resultData ) override;
	void Reorg( const CBlobDesc& source, const CIntHandle& sourceData, int stride, bool isForward,
		const CBlobDesc& result, const CIntHandle& resultData ) override;
	void AddWidthIndex( const CBlobDesc& source, const CFloatHandle& sourceData, bool isForward, const CFloatHandle& result ) override;
	void AddWidthIndex( const CBlobDesc& source, const CIntHandle& sourceData, bool isForward, const CIntHandle& result ) override;
	void AddHeightIndex( const CBlobDesc& source, const CFloatHandle& sourceData, bool isForward, const CFloatHandle& result ) override;
	void AddHeightIndex( const CBlobDesc& source, const CIntHandle& sourceData, bool isForward, const CIntHandle& result ) override;
	CDropoutDesc* InitDropout( float rate, bool isSpatial, bool isBatchwise, const CBlobDesc& input,
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
