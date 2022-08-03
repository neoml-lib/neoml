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

#include <NeoMathEngine/NeoMathEngineDefs.h>
#include <NeoMathEngine/BlobType.h>
#include <NeoMathEngine/MemoryHandle.h>
#include <NeoMathEngine/BlobDesc.h>
#include <NeoMathEngine/SparseMatrixDesc.h>
#include <NeoMathEngine/LookupData.h>
#include <NeoMathEngine/OpenMP.h>
#include <NeoMathEngine/CrtAllocatedObject.h>
#include <NeoMathEngine/PerformanceCounters.h>
#include <NeoMathEngine/CrtAllocatedObject.h>
#include <NeoMathEngine/NeoMathEngineException.h>
#include <NeoMathEngine/ThreadPool.h>
#include <climits>

namespace NeoML {

// Supported activation functions
enum TActivationFunction {
	AF_Linear = 0,
	AF_ELU,
	AF_ReLU,
	AF_LeakyReLU,
	AF_Abs,
	AF_Sigmoid,
	AF_Tanh,
	AF_HardTanh,
	AF_HardSigmoid,
	AF_Power,
	AF_HSwish,
	AF_GELU,
	AF_Exp,
	AF_Log,
	AF_Erf,

	AF_Count
};

// Supported coordinate modes for linear interpolation
// The variables in formula:
//     - scale - size multiplier
//     - x_old - coordinate in array before the interpolation 
//     - x_new - coordinate in array after the interpolation
//     - old_size - size before the transformation
//     - new_size - size after the transformation  (int(ratio * old_size))
enum class TInterpolationCoords : int {
	HalfPixel, // x_old = ( x_new + 0.5 ) / scale - 0.5
	PytorchHalfPixel, // x_old = ( new_size > 1 ) ? ( x_new + 0.5 ) / scale - 0.5 : 0
	AlignCorners, // x_old = x_new * ( old_size - 1) / ( new_size - 1 )
	Asymmetric, // x_old = x_new / scale

	Count
};

// Suppported rounding for coordinates
// Transform linear interpolation into nearest (if set)
enum class TInterpolationRound : int {
	None, // no rounding, keep interpolation linear
	RoundPreferFloor, // round half down
	RoundPreferCeil, // round half up
	Floor, // always floor
	Ceil, // always ceil

	Count
};

// The class provides operations on vectors
class NEOMATHENGINE_API IVectorMathEngine : public CCrtAllocatedObject {
public:
	virtual ~IVectorMathEngine();
	// Copying the second vector values into the first
	virtual void VectorCopy(const CFloatHandle& first, const CConstFloatHandle& second, int vectorSize) = 0;
	virtual void VectorCopy(const CIntHandle& first, const CConstIntHandle& second, int vectorSize) = 0;
	// Broadcasting the copy to new shape
	// additionalWidth = 1 means broadcasting from fromDesc to toDesc
	// additionalWidth != 1 means broadcasting from (*fromDesc, additionalWidth) to (*toDesc, additionalWidth)
	// where (*desc, additionalWidth) is 8-dimensional shape with last dimension equals additionalWidth,
	// channels count of handle must be additionalWidth times bigger than channels count of corresponding desc.
	virtual void BroadcastCopy(const CIntHandle& toHandle, const CConstIntHandle& fromHandle,
		const CBlobDesc& toDesc, const CBlobDesc& fromDesc, int additionalWidth) = 0;
	virtual void BroadcastCopy(const CFloatHandle& toHandle, const CConstFloatHandle& fromHandle,
		const CBlobDesc& toDesc, const CBlobDesc& fromDesc, int additionalWidth) = 0;

	// Filling a vector with the specified value
	virtual void VectorFill(const CFloatHandle& result, float value, int vectorSize) = 0;
	virtual void VectorFill(const CIntHandle& result, int value, int vectorSize) = 0;

	// Filling a vector with a value stored in MathEngine memory
	virtual void VectorFill(const CFloatHandle& result, int vectorSize, const CConstFloatHandle& value) = 0;
	virtual void VectorFill(const CIntHandle& result, int vectorSize, const CConstIntHandle& value) = 0;

	// Converting data type
	virtual void VectorConvert(const CConstFloatHandle& from, const CIntHandle& to, int vectorSize) = 0;
	virtual void VectorConvert(const CConstIntHandle& from, const CFloatHandle& to, int vectorSize) = 0;

	// Filling a vector using the Bernoulli distribution with p being the probability of 1
	// The elements for which the distribution gives 1 are set to the specified value
	virtual void VectorFillBernoulli( const CFloatHandle& result, float p, int vectorSize, float value, int seed ) = 0;

	// Sets to 0 the elements with absolute value smaller than the threshold
	virtual void FilterSmallValues( const CFloatHandle& data, int dataSize, float threshold ) = 0;

	// Calculates the total of all vector elements
	virtual void VectorSum(const CConstFloatHandle& firstHandle, int vectorSize, const CFloatHandle& resultHandle) = 0;
	// The resultHandle is not set to null
	virtual void VectorSumAdd(const CConstFloatHandle& firstHandle, int vectorSize, const CFloatHandle& resultHandle) = 0;
	// Sum of blob elements along dimension with size `dimension`
	virtual void VectorSumAlongDimension(const CConstFloatHandle& firstHandle, int precedingDimension, int dimension,
		int followingDimension, const CFloatHandle& resultHandle) = 0;
	// Cumulative Sum of blob elements along dimension with size `dimension`
	virtual void VectorCumSumAlongDimension(const CConstFloatHandle& firstHandle, int precedingDimension, int dimension,
		int followingDimension, const CFloatHandle& resultHandle, bool reverse) = 0;
	virtual void VectorCumSumAlongDimension(const CConstIntHandle& firstHandle, int precedingDimension, int dimension,
		int followingDimension, const CIntHandle& resultHandle, bool reverse) = 0;
	// Blob `first` is assumed to be a diagonal matrix
	virtual void VectorSumAlongDimensionDiag( const CConstFloatHandle& firstHandle, int precedingDimension, int dimension,
		int followingDimension, const CFloatHandle& resultHandle ) = 0;
	virtual void VectorCumSumAlongDimensionDiag( const CConstFloatHandle& firstHandle, int precedingDimension, int dimension,
		int followingDimension, const CFloatHandle& resultHandle ) = 0;

	// result = (first == second) ? 1.0 : 0.0 elementwise
	virtual void VectorEqual( const CConstIntHandle& firstHandle, const CConstIntHandle& secondHandle,
		const CFloatHandle& resultHandle, int vectorSize ) = 0;
	// result = (first == value) ? 1.0 : 0.0 elementwise
	virtual void VectorEqualValue( const CConstIntHandle& firstHandle,
		const CFloatHandle& resultHandle, int vectorSize, const CConstIntHandle& valueHandle ) = 0;

	// result = max( first, second )
	virtual void VectorMax( const CConstFloatHandle& firstHandle, float secondValue, const CFloatHandle& resultHandle,
		int vectorSize ) = 0;
	virtual void VectorMaxDiff( const CConstFloatHandle& firstHandle, float secondValue, const CFloatHandle& gradHandle,
		int gradHeight, int gradWidth ) = 0;

	// ELU
	virtual void VectorELU(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle,
		int vectorSize, const CConstFloatHandle& alpha) = 0;
	virtual void VectorELUDiff(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
		const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& alpha) = 0;
	virtual void VectorELUDiffOp(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
		const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& alpha) = 0;

	// ReLU (if upperThreshold > 0, ReLU will be limited by upperThreshold)
	virtual void VectorReLU(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize,
		const CConstFloatHandle& upperThresholdHandle) = 0;
	virtual void VectorReLUDiff(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
		const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& upperThresholdHandle) = 0;
	virtual void VectorReLUDiffOp(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
		const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& upperThresholdHandle) = 0;

	// LeakyReLU
	virtual void VectorLeakyReLU( const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle,
		int vectorSize, const CConstFloatHandle& alpha ) = 0;
	virtual void VectorLeakyReLUDiff( const CConstFloatHandle& firstHandle,
		const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle,
		int vectorSize, const CConstFloatHandle& alpha ) = 0;
	virtual void VectorLeakyReLUDiffOp( const CConstFloatHandle& firstHandle,
		const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle,
		int vectorSize, const CConstFloatHandle& alpha ) = 0;

	// H-Swish. f(x) = x * relu6(x + 3) / 6
	virtual void VectorHSwish( const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, 
		int vectorSize ) = 0;
	virtual void VectorHSwishDiff( const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
		const CFloatHandle& resultHandle, int vectorSize ) = 0;

	// max/min value
	// result = max(first, second)
	virtual void VectorEltwiseMax(const CConstFloatHandle& firstHandle,
		const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle, int vectorSize) = 0;
	// result = min(first, second)
	virtual void VectorEltwiseMin(const CConstFloatHandle& firstHandle,
		const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle, int vectorSize) = 0;

	// abs value
	// result = abs(first)
	virtual void VectorAbs(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize) = 0;
	virtual void VectorAbsDiff(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
		const CFloatHandle& resultHandle, int vectorSize) = 0;
	virtual void VectorAbsDiff(const CConstFloatHandle& sourceGradHandle, int gradHeight, int gradWidth,
		const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle) = 0;

	// Hinge function
	virtual void VectorHinge(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize) = 0;
	virtual void VectorHingeDiff(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
		const CFloatHandle& resultHandle, int vectorSize) = 0;

	virtual void VectorSquaredHinge(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize) = 0;
	virtual void VectorSquaredHingeDiff(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
		const CFloatHandle& resultHandle, int vectorSize) = 0;

	// Huber function (quadratic in the middle, linear ends)
	virtual void VectorHuber(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize) = 0;
	virtual void VectorHuberDerivative(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize) = 0;

	// HardTanh function (-1 : x <= -1; x : -1 < x < 1; 1 : x >= 1)
	virtual void VectorHardTanh(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize) = 0;
	virtual void VectorHardTanhDiff(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
		const CFloatHandle& resultHandle, int vectorSize) = 0;
	virtual void VectorHardTanhDiffOp(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
		const CFloatHandle& resultHandle, int vectorSize) = 0;

	// HardSigmoid function (0 : x <= -1; (x + 1) / 2 : -1 < x < 1; 1 : x >= 1)
	virtual void VectorHardSigmoid(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize, 
		const CConstFloatHandle& slopeHandle, const CConstFloatHandle& biasHandle) = 0;
	virtual void VectorHardSigmoidDiff(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
		const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& slopeHandle, const CConstFloatHandle& biasHandle) = 0;
	virtual void VectorHardSigmoidDiffOp(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
		const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& slopeHandle, const CConstFloatHandle& biasHandle) = 0;

	// result = -first
	virtual void VectorNeg(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize) = 0;

	// result = exp(first)
	virtual void VectorExp(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize) = 0;

	// result = log(first)
	virtual void VectorLog( const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle,
		int vectorSize ) = 0;
	// result = first == 0 ? 0 : 1/first
	virtual void VectorLogDiff( const CConstFloatHandle& sourceGradHandle, int sourceGradHeight, int sourceGradWidth,
		const CConstFloatHandle& valueHandle, const CFloatHandle& resultHandle ) = 0;

	// result = -log(first)
	virtual void VectorNegLog(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize) = 0;

	// result = erf(first)
	virtual void VectorErf( const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize ) = 0;

	// Calculates the Kullback-Leibler distance derivative for a Bernoulli distribution using the distribution parameters
	virtual void VectorBernulliKLDerivative(const CConstFloatHandle& estimationHandle,
		const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& target) = 0;

	// Vector addition
	// result = first + second
	virtual void VectorAdd(const CConstFloatHandle& firstHandle,
		const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle, int vectorSize) = 0;
	virtual void VectorAdd( const CConstIntHandle& firstHandle,
		const CConstIntHandle& secondHandle, const CIntHandle& resultHandle, int vectorSize ) = 0;
	virtual void VectorAddValue(const CConstFloatHandle& firstHandle,
		const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& addition) = 0;
	virtual void VectorAddValue( const CConstIntHandle& firstHandle,
		const CIntHandle& resultHandle, int vectorSize, const CConstIntHandle& addition ) = 0;

	// Vector substraction
	// result = first - second
	virtual void VectorSub(const CConstIntHandle& firstHandle,
		const CConstIntHandle& secondHandle, const CIntHandle& resultHandle, int vectorSize) = 0;
	virtual void VectorSub(const CConstFloatHandle& firstHandle,
		const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle, int vectorSize) = 0;
	virtual void VectorSub(const CConstFloatHandle& firstHandle,
		float second, const CFloatHandle& resultHandle, int vectorSize) = 0;
	virtual void VectorSub(float first,
		const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle, int vectorSize) = 0;

	// Multiplies a vector by a number and adds it to another vector
	// result = first + mult * second
	// You may NOT pass the same handle as secondHandle and resultHandle parameters
	virtual void VectorMultiplyAndAdd(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
		const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& multHandle) = 0;
	// result = first - mult * second
	// You may NOT pass the same handle as secondHandle and resultHandle parameters
	virtual void VectorMultiplyAndSub(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
		const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& multHandle) = 0;

	// Multiplies a vector by a number
	// result = first * multiplier
	virtual void VectorMultiply(const CConstFloatHandle& firstHandle,
		const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& multiplierHandle) = 0;
	virtual void VectorMultiply(const CConstIntHandle& firstHandle,
		const CIntHandle& resultHandle, int vectorSize, const CConstIntHandle& multiplierHandle) = 0;
	// result = -first * multiplier
	virtual void VectorNegMultiply(const CConstFloatHandle& firstHandle,
		const CFloatHandle& resultHandle, int vectorSize, const CConstFloatHandle& multiplierHandle) = 0;

	// result = first * second elementwise
	virtual void VectorEltwiseMultiply( const CConstIntHandle& firstHandle,
		const CConstIntHandle& secondHandle, const CIntHandle& resultHandle, int vectorSize ) = 0;
	virtual void VectorEltwiseMultiply(const CConstFloatHandle& firstHandle,
		const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle, int vectorSize) = 0;
	// result += first * second elementwise
	virtual void VectorEltwiseMultiplyAdd(const CConstFloatHandle& firstHandle,
		const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle, int vectorSize) = 0;
	// result = -first * second elementwise
	virtual void VectorEltwiseNegMultiply(const CConstFloatHandle& firstHandle,
		const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle, int vectorSize) = 0;

	// result = first / second elementwise
	virtual void VectorEltwiseDivide( const CConstIntHandle& firstHandle,
		const CConstIntHandle& secondHandle, const CIntHandle& resultHandle, int vectorSize ) = 0;
	virtual void VectorEltwiseDivide(const CConstFloatHandle& firstHandle,
		const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle, int vectorSize) = 0;

	// result = first ^ second
	virtual void VectorEltwisePower(const CConstFloatHandle& firstHandle,
		const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle, int vectorSize) = 0;

	// result = sqrt(first)
	virtual void VectorSqrt(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize) = 0;

	// result = 1 / first
	virtual void VectorInv(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize) = 0;

	// result = min(max(first, minValue), maxValue)
	virtual void VectorMinMax(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize,
		const CConstFloatHandle& minHandle, const CConstFloatHandle& maxHandle) = 0;
	virtual void VectorMinMaxDiff(const CConstFloatHandle& sourceGradHandle, int gradHeight, int gradWidth,
		const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle,
		const CConstFloatHandle& minHandle, const CConstFloatHandle& maxHandle) = 0;

	virtual void VectorSigmoid(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize) = 0;
	// resultHandle = sigmoid-derivative(firstHandle) * secondHandle
	virtual void VectorSigmoidDiff(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
		const CFloatHandle& resultHandle, int vectorSize) = 0;
	// resultHandle = sigmoid-derivative(sigmoid-opposite(firstHandle)) * secondHandle
	virtual void VectorSigmoidDiffOp(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
		const CFloatHandle& resultHandle, int vectorSize) = 0;

	virtual void VectorTanh(const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize) = 0;
	// resultHandle = tanh-derivative(firstHandle) * secondHandle
	virtual void VectorTanhDiff(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
		const CFloatHandle& resultHandle, int vectorSize) = 0;
	// resultHandle = tanh-derivative(tanh-opposite(firstHandle)) * secondHandle
	virtual void VectorTanhDiffOp(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
		const CFloatHandle& resultHandle, int vectorSize) = 0;

	virtual void VectorPower(float exponent, const CConstFloatHandle& firstHandle, const CFloatHandle& resultHandle, int vectorSize) = 0;
	virtual void VectorPowerDiff(float exponent, const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
		const CFloatHandle& resultHandle, int vectorSize) = 0;
	virtual void VectorPowerDiffOp(float exponent, const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
		const CFloatHandle& resultHandle, int vectorSize) = 0;

	// L1-regularization (adding a scaled gradient of L1-regularizer). L1 is represented by a Hubert function
	// result = first + mult * L1Diff(hubertThreshold, second), where mult and hubertThreshold have scalar values
	virtual void VectorL1DiffAdd(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
		const CFloatHandle& resultHandle, int vectorSize,
		const CConstFloatHandle& hubertThresholdHandle, const CConstFloatHandle& multHandle) = 0;

	virtual void VectorDotProduct(const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle, int vectorSize,
		const CFloatHandle& resultHandle) = 0;

	// result[i] = first[i] == 0 ? 1 : 0
	virtual void VectorEltwiseNot( const CConstIntHandle& firstHandle, const CIntHandle& resultHandle, int vectorSize ) = 0;

	// result[i] = first[i] >= 0 ? 1.f : 0.f
	virtual void VectorEltwiseNotNegative( const CConstIntHandle& firstHanle, const CFloatHandle& resultHandle, int vectorSize ) = 0;

	// result[i] = first[i] < second[i] ? 1.f : 0.f
	virtual void VectorEltwiseLess( const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
		const CFloatHandle& resultHandle, int vectorSize ) = 0;
	virtual void VectorEltwiseLess( const CConstFloatHandle& firstHandle, float second,
		const CFloatHandle& resultHandle, int vectorSize ) = 0;
	virtual void VectorEltwiseLess( float firstHandle, const CConstFloatHandle& secondHandle,
		const CFloatHandle& resultHandle, int vectorSize ) = 0;

	// result[i] = first[i] < second[i] ? 1 : 0
	virtual void VectorEltwiseLess( const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
		const CIntHandle& resultHandle, int vectorSize ) = 0;
	virtual void VectorEltwiseLess( const CConstIntHandle& firstHandle, const CConstIntHandle& secondHandle,
		const CIntHandle& resultHandle, int vectorSize ) = 0;

	// result[i] = first[i] == second[i] ? 1 : 0
	virtual void VectorEltwiseEqual( const CConstFloatHandle& firstHandle, const CConstFloatHandle& secondHandle,
		const CIntHandle& resultHandle, int vectorSize ) = 0;
	virtual void VectorEltwiseEqual( const CConstIntHandle& firstHandle, const CConstIntHandle& secondHandle,
		const CIntHandle& resultHandle, int vectorSize ) = 0;

	// result[i] = first[i] != 0 ? second[i] : third[i]
	virtual void VectorEltwiseWhere( const CConstIntHandle& firstHandle, const CConstFloatHandle& secondHandle,
		const CConstFloatHandle& thirdHandle, const CFloatHandle& resultHandle, int vectorSize ) = 0;
	virtual void VectorEltwiseWhere( const CConstIntHandle& firstHandle, const CConstIntHandle& secondHandle,
		const CConstIntHandle& thirdHandle, const CIntHandle& resultHandle, int vectorSize ) = 0;

	virtual void VectorFindMaxValueInSet(const CConstFloatHandle* vectors, int vectorCount, const CFloatHandle& resultHandle, int vectorSize) = 0;
	virtual void VectorFindMaxValueInSet(const CConstFloatHandle* vectors, int vectorCount, const CFloatHandle& resultHandle,
		const CIntHandle& indexHandle, int vectorSize) = 0;
	virtual void VectorSpreadValues(const CConstFloatHandle& sourceHandle, CFloatHandle* vectors, int vectorCount,
		const CConstIntHandle& indexHandle, int vectorSize) = 0;

	virtual void VectorTopK(const CConstFloatHandle& first, int firstSize, int k, const CFloatHandle& result, const CIntHandle& indices) = 0;

	virtual void VectorTopKDiff(const CConstFloatHandle& sourceGrad, int sourceGradHeight, int sourceGradWidth,
		const CConstIntHandle& indices, int k, const CFloatHandle& resultGrad) = 0;
};

//------------------------------------------------------------------------------------------------------------

// The class provides basic linear algebra operations
class NEOMATHENGINE_API IBlasEngine : public IVectorMathEngine {
public:
	virtual ~IBlasEngine();
	// Fill each of a matrix rows with the specified vector
	virtual void SetVectorToMatrixRows(const CFloatHandle& resultHandle, int matrixHeight,
		int matrixWidth, const CConstFloatHandle& vectorHandle) = 0;

	// Adds the vector elements to the matrix elements; the indices vector specifies to which element in the row
	virtual void AddVectorToMatrixElements(const CFloatHandle& matrix, int height, int width,
		const CConstIntHandle& indices, const CConstFloatHandle& vector) = 0;
	// Adds the vector elements to the matrix elements; the indices in both rows and columns are specified
	virtual void AddVectorToMatrixElements(const CFloatHandle& matrix, int height, int width,
		const CConstIntHandle& rowIndices, const CConstIntHandle& columnIndices,
		const CConstFloatHandle& vector, int vectorSize) = 0;
	// Reverse functions: adding the specified matrix elements to the vector
	virtual void AddMatrixElementsToVector(const CConstFloatHandle& matrix, int height, int width,
		const CConstIntHandle& indices, const CFloatHandle& result, int vectorSize) = 0;
	virtual void AddMatrixElementsToVector(const CConstFloatHandle& matrix, int height, int width,
		const CConstIntHandle& rowIndices, const CConstIntHandle& columnIndices,
		const CFloatHandle& result, int vectorSize) = 0;

	// Elementwise adds two matrices of the same size
	virtual void AddDiagMatrixToMatrix( const CConstFloatHandle& diagMatrix, const CConstFloatHandle& matrix,
		int height, int width, const CFloatHandle& result ) = 0;

	// Elementwise adds two matrices of the same size
	virtual void AddMatrixElementsToMatrix(const CConstFloatHandle& matrix, int height, int width,
		const CFloatHandle& result, const CConstIntHandle& indices) = 0;

	// Adds the specified vector to each row of the matrix
	virtual void AddVectorToMatrixRows(int batchSize, const CConstFloatHandle& matrixHandle, const CFloatHandle& resultHandle,
		int matrixHeight, int matrixWidth, const CConstFloatHandle& vectorHandle) = 0;

	// Adds/subtracts the specified vector to each column of the matrix
	virtual void AddVectorToMatrixColumns(const CConstFloatHandle& matrixHandle, const CFloatHandle& resultHandle,
		int matrixHeight, int matrixWidth, const CConstFloatHandle& vectorHandle) = 0;
	virtual void AddVectorToMatrixColumns( const CConstIntHandle& matrixHandle, const CIntHandle& resultHandle,
		int matrixHeight, int matrixWidth, const CConstIntHandle& vectorHandle ) = 0;
	virtual void SubVectorFromMatrixColumns(const CConstFloatHandle& matrixHandle, const CFloatHandle& resultHandle,
		int matrixHeight, int matrixWidth, const CConstFloatHandle& vectorHandle) = 0;

	// Calculates the total of matrix rows
	virtual void SumMatrixRowsAdd(int batchSize, const CFloatHandle& resultHandle, const CConstFloatHandle& matrixHandle,
		int matrixHeight, int matrixWidth) = 0;
	virtual void SumMatrixRows(int batchSize, const CFloatHandle& resultHandle, const CConstFloatHandle& matrixHandle,
		int matrixHeight, int matrixWidth) = 0;
	virtual void SumMatrixRows(int batchSize, const CIntHandle& resultHandle, const CConstIntHandle& matrixHandle,
		int matrixHeight, int matrixWidth) = 0;

	// Calculates the total of matrix columns
	virtual void SumMatrixColumns(const CFloatHandle& resultHandle, const CConstFloatHandle& matrixHandle,
		int matrixHeight, int matrixWidth) = 0;

	// Elementwise divide matrix columns by vector of size matrixHeight.
	virtual void MatrixColumnsEltwiseDivide( const CConstFloatHandle& matrix, int matrixHeight, int matrixWidth,
		const CConstFloatHandle& vector, const CFloatHandle& resultHandle ) = 0;

	// Vector operations over matrix rows
	// log(exp(x0) + ... + exp(xn)), the result is a vector with "height" elements
	virtual void MatrixLogSumExpByRows(const CConstFloatHandle& matrix, int height, int width, const CFloatHandle& result, int resultSize) = 0;
	// softmax : exp(xi) / (exp(x0) + ... + exp(xn))
	virtual void MatrixSoftmaxByRows(const CConstFloatHandle& matrix, int height, int width, const CFloatHandle& result) = 0;
	// Calculates the derivative of softmax, 
	// expressed in terms of softmax elementwise multiplied by the "second" parameter (used for in-place backpropagation)
	virtual void MatrixSoftmaxDiffOpByRows(const CConstFloatHandle& first, const CConstFloatHandle& second,
		int height, int width, const CFloatHandle& result) = 0;

	// Vector operations over matrix columns
	// softmax : exp(xi) / (exp(x0) + ... + exp(xn))
	virtual void MatrixSoftmaxByColumns(const CConstFloatHandle& matrix, int height, int width,
		const CFloatHandle& result) = 0;
	// Calculates the derivative of softmax,
	// expressed in terms of softmax elementwise multiplied by the "second" parameter (used for in-place backpropagation)
	virtual void MatrixSoftmaxDiffOpByColumns(const CConstFloatHandle& first, const CConstFloatHandle& second,
		int height, int width, const CFloatHandle& result) = 0;

	// Finds the maximum value in each matrix row
	virtual void FindMaxValueInRows(const CConstFloatHandle& matrixHandle, int matrixHeight, int matrixWidth,
		const CFloatHandle& resultHandle, const CIntHandle& columnIndices, int vectorSize) = 0;
	virtual void FindMaxValueInRows(const CConstFloatHandle& matrixHandle,
		int matrixHeight, int matrixWidth, const CFloatHandle& resultHandle, int vectorSize) = 0;

	// Finds the maximum value in each matrix column, for a set of batchSize matrices, matrixHeight * matrixWidth each
	// The result is of batchSize * matrixWidth size
	virtual void FindMaxValueInColumns( int batchSize, const CConstFloatHandle& matrixHandle, int matrixHeight,
		int matrixWidth, const CFloatHandle& resultHandle, const CIntHandle& rowIndices, int vectorSize ) = 0;

	// Finds the minimum value in each matrix column
	virtual void FindMinValueInColumns( const CConstFloatHandle& matrixHandle, int matrixHeight, int matrixWidth,
		const CFloatHandle& resultHandle, const CIntHandle& columnIndices ) = 0;

	// Performs multichannel lookup; each channel has its own vector representation table
	// When the vector representation is found in the table, it is copied to the output
	// If there are fewer tables than channels, the contents of extra channels are copied to the output as is
	virtual void VectorMultichannelLookupAndCopy(int batchSize, int channelCount, const CConstFloatHandle& inputHandle,
		const CConstFloatHandle* lookupHandles, const CLookupDimension* lookupDimensions, int lookupCount, 
		const CFloatHandle& outputHandle, int outputChannels) = 0;
	virtual void VectorMultichannelLookupAndCopy(int batchSize, int channelCount, const CConstIntHandle& inputHandle,
		const CConstFloatHandle* lookupHandles, const CLookupDimension* lookupDimensions, int lookupCount,
		const CFloatHandle& outputHandle, int outputChannels) = 0;
	virtual void VectorMultichannelLookupAndCopy(int batchSize, int channelCount, const CConstIntHandle& inputHandle,
		const CConstIntHandle* lookupHandles, const CLookupDimension* lookupDimensions, int lookupCount,
		const CIntHandle& outputHandle, int outputChannels) = 0;
	// Finds the position in the representation table for the channel and adds a row from the specified matrix (of batchSize height)
	virtual void VectorMultichannelLookupAndAddToTable(int batchSize, int channelCount, const CConstFloatHandle& inputHandle,
		const CFloatHandle* lookupHandles, const CLookupDimension* lookupDimensions, int lookupCount, 
		const CConstFloatHandle& multHandle, const CConstFloatHandle& matrixHandle, int outputChannels) = 0;
	virtual void VectorMultichannelLookupAndAddToTable(int batchSize, int channelCount, const CConstIntHandle& inputHandle,
		const CFloatHandle* lookupHandles, const CLookupDimension* lookupDimensions, int lookupCount,
		const CConstFloatHandle& multHandle, const CConstFloatHandle& matrixHandle, int outputChannels) = 0;

	virtual void LookupAndSum( const CConstIntHandle& indicesHandle, int batchSize, int indexCount,
		const CConstFloatHandle& tableHandle, int vectorSize, const CFloatHandle& result ) = 0;
	virtual void LookupAndAddToTable( const CConstIntHandle& indicesHandle, int batchSize, int indexCount,
		const CConstFloatHandle& additionsHandle, int vectorSize, const CFloatHandle& tableHandle, int vectorCount ) = 0;

	// Binarizes an enumeration
	virtual void EnumBinarization(int batchSize, const CConstFloatHandle& inputHandle, int enumSize,
		const CFloatHandle& resultHandle) = 0;
	virtual void EnumBinarization(int batchSize, const CConstIntHandle& inputHandle, int enumSize,
		const CFloatHandle& resultHandle) = 0;

	// Binarizes a bit set. Each bit of the input is represented by a float number in the output. bitSetSize means the number of int values
	// outputVectorSize may be less than bitSetSize * 32
	virtual void BitSetBinarization(int batchSize, int bitSetSize,
		const CConstIntHandle& inputHandle, int outputVectorSize, const CFloatHandle& resultHandle) = 0;

	// Working with lookup matrices
	// Multiplies a batch of matrices stored one after another by a batch of vectors stored one after another
	virtual void MultiplyLookupMatrixByLookupVector(int batchSize, const CLookupMatrix& matrix,
		const CLookupVector& vector, const CFloatHandle& result, int resultSize) = 0;
	virtual void MultiplyTransposedLookupMatrixByVector(int batchSize, const CLookupMatrix& matrix,
		const CConstFloatHandle& vectorHandle, const CFloatHandle& resultHandle, int resultSize) = 0;
	virtual void MultiplyTransposedLookupMatrixByVectorAndAdd(int batchSize, const CLookupMatrix& matrix,
		const CConstFloatHandle& vectorHandle, const CFloatHandle& resultHandle, int resultSize) = 0;
	virtual void MultiplyVectorByTransposedLookupVectorAndAddToTable(int batchSize,
		const CFloatHandle& tableHandle, int vectorCount, int vectorSize, const CConstIntHandle& indices,
		const CConstFloatHandle& firstHandle, int firstSize, const CLookupVector& secondVector) = 0;

	// Multiplies a matrix by another matrix, transposed; the result will be of firstHeight * secondHeight size
	virtual void MultiplyMatrixByTransposedMatrix(const CConstFloatHandle& firstHandle, int firstHeight,
		int firstWidth, int firstRowSize, const CConstFloatHandle& secondHandle, int secondHeight, int secondRowSize,
		const CFloatHandle& resultHandle, int resultRowSize, int resultBufferSize) = 0;
	// Multiplies matrices from two batches, stored one after another in firstHandle, secondHandle parameters
	virtual void MultiplyMatrixByTransposedMatrix(int batchSize, const CConstFloatHandle& firstHandle, int firstHeight,
		int firstWidth, const CConstFloatHandle& secondHandle, int secondHeight, const CFloatHandle& resultHandle,
		int resultBufferSize) = 0;

	// Operations on sparse matrices

	// result = first * T(second). The result will be of firstHeight * secondHeight size
	// IMPORTANT: when working on CPU the resultHandle should not contain inf or nan (due to a bug in MKL).
	virtual void MultiplySparseMatrixByTransposedMatrix( int firstHeight, int firstWidth, int secondHeight,
		const CSparseMatrixDesc& firstDesc, const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle ) = 0;

	// result = T(first) * second, second is transposed if isTransposedSparse
	virtual void MultiplyTransposedMatrixBySparseMatrix( int firstHeight, int firstWidth, int resultWidth,
		const CConstFloatHandle& firstHandle, const CSparseMatrixDesc& secondDesc, const CFloatHandle& resultHandle,
		bool isTransposedSparse ) = 0;

	// result = result + T(first) * second. The result will be of firstWidth * secondWidth size
	virtual void MultiplyTransposedMatrixBySparseMatrixAndAdd( int firstHeight, int firstWidth, int secondWidth,
		const CConstFloatHandle& firstHandle, const CSparseMatrixDesc& secondDesc, const CFloatHandle& resultHandle ) = 0;

	// result = first * second. The result will be of firstHeight * secondWidth size
	virtual void MultiplySparseMatrixByMatrix( int firstHeight, int firstWidth, int secondWidth,
		const CSparseMatrixDesc& firstDesc, const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle ) = 0;

	// result = T(first) * second. The result size is firstWidth * secondWidth
	virtual void MultiplyTransposedSparseMatrixByMatrix( int firstHeight, int firstWidth, int secondWidth,
		const CSparseMatrixDesc& firstDesc, const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle ) = 0;

	// result = result + first(T) * second
	virtual void MultiplyTransposedMatrixByMatrixAndAdd(const CConstFloatHandle& firstHandle, int firstHeight, int firstWidth, int firstRowSize,
		const CConstFloatHandle& secondHandle, int secondWidth, int secondRowSize,
		const CFloatHandle& resultHandle, int resultRowSize, int resultBufferSize) = 0;

	// result[i] = first[i](T) * second[i] for i in [0, batchSize)
	virtual void MultiplyTransposedMatrixByMatrix(int batchSize, const CConstFloatHandle& firstHandle, int firstHeight, int firstWidth,
		const CConstFloatHandle& secondHandle, int secondWidth, const CFloatHandle& resultHandle, int resultBufferSize) = 0;

	virtual void MultiplyDiagMatrixByMatrix(const CConstFloatHandle& firstHandle, int firstSize,
		const CConstFloatHandle& secondHandle, int secondWidth,
		const CFloatHandle& resultHandle, int resultBufferSize) = 0;
	// Mutliplies diagonal matrices as in MultiplyDiagMatrixByMatrix, 
	// for a set of matrices stored one after another in "second" and "result" parameters; "first" parameter is the same
	virtual void Multiply1DiagMatrixByMatrix(int batchSize, const CConstFloatHandle& firstHandle, int firstSize,
		const CConstFloatHandle& secondHandle, int secondWidth,
		const CFloatHandle& resultHandle, int resultBufferSize) = 0;

	// Multiplies matrices from two batches, stored one after another in firstHandle, secondHandle parameters
	virtual void MultiplyMatrixByMatrix( int batchSize, const CConstFloatHandle& firstHandle, int firstHeight,
		int firstWidth, const CConstFloatHandle& secondHandle, int secondWidth,
		const CFloatHandle& resultHandle, int resultBufferSize ) = 0;

	virtual void MultiplyMatrixByDiagMatrix(const CConstFloatHandle& firstHandle, int firstHeight, int firstWidth,
		const CConstFloatHandle& secondHandle, const CFloatHandle& resultHandle, int resultBufferSize) = 0;

	// Transposes a set of matrices stored one after another. A matrix cell is of "channels" size
	virtual void TransposeMatrix(int batchSize, const CConstFloatHandle& firstHandle,
		int height, int medium, int width, int channels, const CFloatHandle& resultHandle, int resultBufferSize) = 0;

	virtual void TransposeMatrix(int batchSize, const CConstIntHandle& firstHandle,
		int height, int medium, int width, int channels, const CIntHandle& resultHandle, int resultBufferSize) = 0;

	// Adds up the products of diagonal matrices from firstHandle (of "firstSize" size)
	// to matrices from secondHandle (of firstSize * secondWidth size)
	virtual void MultiplyDiagMatrixByMatrixAndAdd(int batchSize, const CConstFloatHandle& firstHandle,
		int firstSize, const CConstFloatHandle& secondHandle, int secondWidth, const CFloatHandle& resultHandle) = 0;

	// Calculates the dot product of matrix rows for two matrices of the same size
	virtual void RowMultiplyMatrixByMatrix( const CConstFloatHandle& firstHandle,
		const CConstFloatHandle& secondHandle, int height, int width, const CFloatHandle& result ) = 0;

	// "Spreads" the matrix rows into another matrix according to the specified indices
	// The empty rows are filled with fillValue or 0 if fillValue is null
	virtual void MatrixSpreadRows( const CConstFloatHandle& sourceHandle, int height, int width,
		const CFloatHandle& resultHandle, int resultHeight, const CConstIntHandle& indexHandle,
		const CConstFloatHandle& fillValue ) = 0;
	virtual void MatrixSpreadRowsAdd( const CConstFloatHandle& sourceHandle, int height, int width,
		const CFloatHandle& resultHandle, int resultHeight, const CConstIntHandle& indexHandle ) = 0;
	virtual void MatrixSpreadRows( const CConstIntHandle& sourceHandle, int height, int width,
		const CIntHandle& resultHandle, int resultHeight, const CConstIntHandle& indexHandle,
		const CConstIntHandle& fillValue ) = 0;

	// Computes the singular value decomposition of the dense matrix `a`: `a` = `u` * `s` * `vt`
	virtual void SingularValueDecomposition( const CFloatHandle& a, int height, int width, const CFloatHandle& u, const CFloatHandle& s,
		const CFloatHandle& vt, const CFloatHandle& superb, bool returnLeftVectors, bool returnRightVectors ) = 0;

	// Computes the QR factorization of the matrix = Q x R, where Q is of shape (height, min(height, width)), R is of shape(height, width).
	// If returnQ or returnR is false then the corresponding matrix in the factorization is not returned.
	// If inplace is true then the matrixHandle will be overwritten with a content of matrix R.
	virtual void QRFactorization( int height, int width, const CFloatHandle& matrixHandle, const CFloatHandle* qHandle, const CFloatHandle* rHandle,
		bool inplace, bool returnQ, bool returnR ) = 0;
};

// Blob operations descriptors
struct NEOMATHENGINE_API CTimeConvolutionDesc : public CCrtAllocatedObject { public: virtual ~CTimeConvolutionDesc(); };
struct NEOMATHENGINE_API C3dConvolutionDesc : public CCrtAllocatedObject { public: virtual ~C3dConvolutionDesc(); };
struct NEOMATHENGINE_API CConvolutionDesc : public CCrtAllocatedObject { public: virtual ~CConvolutionDesc(); };
struct NEOMATHENGINE_API CChannelwiseConvolutionDesc : public CCrtAllocatedObject { public: virtual ~CChannelwiseConvolutionDesc(); };
struct NEOMATHENGINE_API CRleConvolutionDesc : public CCrtAllocatedObject { public: virtual ~CRleConvolutionDesc(); };
struct NEOMATHENGINE_API CDropoutDesc : public CCrtAllocatedObject { public: virtual ~CDropoutDesc(); };
struct NEOMATHENGINE_API CGlobalMaxPoolingDesc : public CCrtAllocatedObject { public: virtual ~CGlobalMaxPoolingDesc(); };
struct NEOMATHENGINE_API CMaxPoolingDesc : public CCrtAllocatedObject { public: virtual ~CMaxPoolingDesc(); };
struct NEOMATHENGINE_API CMeanPoolingDesc : public CCrtAllocatedObject { public: virtual ~CMeanPoolingDesc(); };
struct NEOMATHENGINE_API C3dMaxPoolingDesc : public CCrtAllocatedObject { public: virtual ~C3dMaxPoolingDesc(); };
struct NEOMATHENGINE_API C3dMeanPoolingDesc : public CCrtAllocatedObject { public: virtual ~C3dMeanPoolingDesc(); };
struct NEOMATHENGINE_API CGlobalMaxOverTimePoolingDesc : public CCrtAllocatedObject { public: virtual ~CGlobalMaxOverTimePoolingDesc(); };
struct NEOMATHENGINE_API CMaxOverTimePoolingDesc : public CCrtAllocatedObject { public: virtual ~CMaxOverTimePoolingDesc(); };
struct NEOMATHENGINE_API CLrnDesc : public CCrtAllocatedObject { public: virtual ~CLrnDesc(); };
struct NEOMATHENGINE_API CLstmDesc : public CCrtAllocatedObject { public: virtual ~CLstmDesc(); };

//------------------------------------------------------------------------------------------------------------
// RLE format

static const int MaxRleConvFilterWidth = 16;
static const int MaxRleConvImageWidth = 64;

// RLE stroke
struct CRleStroke {
	short Start;	// stroke start
	short End;		// stroke end (the first pixel NOT in the stroke)

	static CRleStroke Sentinel() { return { SHRT_MAX, -1 }; }
};

static const CRleStroke Sentinel = { SHRT_MAX, -1 };

struct CRleImage {
	int StrokesCount;
	int Height;
	int Width;
	CRleStroke Stub;
	CRleStroke Lines[1];
};

//------------------------------------------------------------------------------------------------------------

// Neural network-specific operations
class NEOMATHENGINE_API IDnnEngine : public IBlasEngine {
public:
	virtual ~IDnnEngine();
	// Blob operations
	// Operations that use the dimension specified by name
	virtual void BlobMergeByDim( TBlobDim dim, const CBlobDesc* from, const CFloatHandle* fromData, int fromCount,
		const CBlobDesc& to, const CFloatHandle& toData ) = 0;
	virtual void BlobMergeByDim( TBlobDim dim, const CBlobDesc* from, const CIntHandle* fromData, int fromCount,
		const CBlobDesc& to, const CIntHandle& toData ) = 0;
	virtual void BlobSplitByDim( TBlobDim dim, const CBlobDesc& from, const CFloatHandle& fromData,
		const CBlobDesc* to, const CFloatHandle* toData, int toCount ) = 0;
	virtual void BlobSplitByDim( TBlobDim dim, const CBlobDesc& from, const CIntHandle& fromData,
		const CBlobDesc* to, const CIntHandle* toData, int toCount ) = 0;

	// Resizes images in a blob
	// For delta < 0 the pixels at the edges are erased
	// For delta > 0 the extra pixels at the edges are filled with defaultValue
	virtual void BlobResizeImage( const CBlobDesc& from, const CFloatHandle& fromData, int deltaLeft, int deltaRight,
		int deltaTop, int deltaBottom, float defaultValue, const CBlobDesc& to, const CFloatHandle& toData ) = 0;

	// Retrieves subsequences from the blob sequences and, if necessary, reverses them
	virtual void BlobGetSubSequence( const CBlobDesc& from, const CFloatHandle& fromData, const CIntHandle& indexHandle,
		const CBlobDesc& to, const CFloatHandle& toData, int startPos, bool isRev ) = 0;

	// Time convolution
	// The descriptor should be destroyed using the standard delete operator after use.
	virtual CTimeConvolutionDesc* InitTimeConvolution( const CBlobDesc& source, int stride, int paddingFront,
		int paddingBack, int dilation, const CBlobDesc& filter, const CBlobDesc& result ) = 0;

	virtual void BlobTimeConvolution( const CTimeConvolutionDesc& desc, const CConstFloatHandle& source,
		const CConstFloatHandle& filter, const CConstFloatHandle& freeTerm, const CFloatHandle& result ) = 0;
	virtual void BlobTimeConvolutionBackward( const CTimeConvolutionDesc& desc, const CConstFloatHandle& outputDiff,
		const CConstFloatHandle& filter, const CConstFloatHandle& freeTerm, const CFloatHandle& inputDiff ) = 0;
	virtual void BlobTimeConvolutionLearnAdd( const CTimeConvolutionDesc& desc, const CConstFloatHandle& input,
		const CConstFloatHandle& outputDiff, const CFloatHandle& filterDiff, const CFloatHandle& freeTermDiff ) = 0;

	// 3D convolution
	// The descriptor should be destroyed using the standard delete operator after use.
	virtual C3dConvolutionDesc* InitBlob3dConvolution( const CBlobDesc& input,
		int paddingHeight, int paddingWidth, int paddingDepth,
		int strideHeight, int strideWidth, int strideDepth,
		const CBlobDesc& filter, const CBlobDesc& output ) = 0;

	virtual void Blob3dConvolution( const C3dConvolutionDesc& desc, const CConstFloatHandle& source,
		const CConstFloatHandle& filter, const CConstFloatHandle* freeTerm, const CFloatHandle& result ) = 0;
	virtual void Blob3dConvolutionBackward( const C3dConvolutionDesc& desc, const CConstFloatHandle& source,
		const CConstFloatHandle& filter, const CConstFloatHandle* freeTerm, const CFloatHandle& result ) = 0;
	virtual void Blob3dConvolutionLearnAdd( const C3dConvolutionDesc& desc, const CConstFloatHandle& input,
		const CConstFloatHandle& outputDiff, const CFloatHandle& filterDiff,
		const CFloatHandle* freeTermDiff, bool isFreeTermDiffFromInput ) = 0;

	// Convolution
	// The descriptor should be destroyed using the standard delete operator after use.
	virtual CConvolutionDesc* InitBlobConvolution( const CBlobDesc& input, int paddingHeight, int paddingWidth,
		int strideHeight, int strideWidth, int dilationHeight, int dilationWidth, const CBlobDesc& filter,
		const CBlobDesc& output ) = 0;

	virtual void BlobConvolution( const CConvolutionDesc& desc, const CConstFloatHandle& source,
		const CConstFloatHandle& filter, const CConstFloatHandle* freeTerm, const CFloatHandle& result ) = 0;
	virtual void BlobConvolutionBackward( const CConvolutionDesc& desc, const CConstFloatHandle& outputDiff,
		const CConstFloatHandle& filter, const CConstFloatHandle* freeTerm, const CFloatHandle& inputDiff ) = 0;
	virtual void BlobConvolutionLearnAdd( const CConvolutionDesc& desc, const CConstFloatHandle& input,
		const CConstFloatHandle& outputDiff, const CFloatHandle& filterDiff,
		const CFloatHandle* freeTermDiff, bool isFreeTermDiffFromInput ) = 0;

	// Calculates channelwise convolution
	// You can pass 0 for the freeTerm parameter, and the free terms will be 0
	// The descriptor should be destroyed using the standard delete operator after use.
	virtual CChannelwiseConvolutionDesc* InitBlobChannelwiseConvolution( 
		const CBlobDesc& input, int paddingHeight, int paddingWidth, int strideHeight, int strideWidth,
		const CBlobDesc& filter, const CBlobDesc* freeTerm, const CBlobDesc& output ) = 0;

	virtual void BlobChannelwiseConvolution( const CChannelwiseConvolutionDesc& desc, const CConstFloatHandle& source,
		const CConstFloatHandle& filter, const CConstFloatHandle* freeTerm, const CFloatHandle& result ) = 0;
	// Calculates the derivative by the input for the channelwise convolution 
	// when the derivative by the output is known
	virtual void BlobChannelwiseConvolutionBackward( const CChannelwiseConvolutionDesc& desc,
		const CConstFloatHandle& source, const CConstFloatHandle& filter, const CFloatHandle& result ) = 0;
	// Calculates the derivative by parameters for the channelwise convolution
	// when the input and the derivative by the output are known
	virtual void BlobChannelwiseConvolutionLearnAdd( const CChannelwiseConvolutionDesc& desc,
		const CConstFloatHandle& input, const CConstFloatHandle& outputDiff, const CFloatHandle& filterDiff,
		const CFloatHandle* freeTermDiff ) = 0;

	// GlobalMaxPooling
	// The descriptor should be destroyed using the standard delete operator after use.
	virtual CGlobalMaxPoolingDesc* InitGlobalMaxPooling( const CBlobDesc& source, const CBlobDesc& maxIndices,
		const CBlobDesc& result ) = 0;

	virtual void BlobGlobalMaxPooling( const CGlobalMaxPoolingDesc& desc,
		const CConstFloatHandle& source, const CIntHandle& maxIndices, const CFloatHandle& result ) = 0;
	virtual void BlobGlobalMaxPoolingBackward( const CGlobalMaxPoolingDesc& desc,
		const CFloatHandle& outputDiff, const CIntHandle& maxIndices, const CFloatHandle& inputDiff ) = 0;

	// 3dMax-Pooling
	// The descriptor should be destroyed using the standard delete operator after use.
	virtual C3dMaxPoolingDesc* Init3dMaxPooling( const CBlobDesc& source,
		int filterHeight, int filterWidth, int filterDepth,
		int strideHeight, int strideWidth, int strideDepth, const CBlobDesc& result ) = 0;

	virtual void Blob3dMaxPooling( const C3dMaxPoolingDesc& desc, const CFloatHandle& source,
		const CIntHandle* maxIndices, const CFloatHandle& result ) = 0;
	virtual void Blob3dMaxPoolingBackward( const C3dMaxPoolingDesc& desc, const CFloatHandle& outputDiff,
		const CIntHandle& maxIndices, const CFloatHandle& inputDiff ) = 0;

	// 3dMean-Pooling
	// The descriptor should be destroyed using the standard delete operator after use.
	virtual C3dMeanPoolingDesc* Init3dMeanPooling( const CBlobDesc& source,
		int filterHeight, int filterWidth, int filterDepth,
		int strideHeight, int strideWidth, int strideDepth,
		const CBlobDesc& result ) = 0;

	virtual void Blob3dMeanPooling( const C3dMeanPoolingDesc& desc, const CFloatHandle& source,
		const CFloatHandle& result ) = 0;
	virtual void Blob3dMeanPoolingBackward( const C3dMeanPoolingDesc& desc, const CFloatHandle& outputDiff,
		const CFloatHandle& inputDiff) = 0;

	// Max-Pooling
	// The descriptor should be destroyed using the standard delete operator after use.
	virtual CMaxPoolingDesc* InitMaxPooling( const CBlobDesc& source,
		int filterHeight, int filterWidth, int strideHeight, int strideWidth,
		const CBlobDesc& result ) = 0;

	virtual void BlobMaxPooling( const CMaxPoolingDesc& desc, const CFloatHandle& source,
		const CIntHandle* maxIndices, const CFloatHandle& result) = 0;
	virtual void BlobMaxPoolingBackward( const CMaxPoolingDesc& desc, const CFloatHandle& outputDiff,
		const CIntHandle& maxIndices, const CFloatHandle& inputDiff) = 0;

	// MaxOverTime-Pooling
	// The descriptor should be destroyed using the standard delete operator after use.
	virtual CMaxOverTimePoolingDesc* InitMaxOverTimePooling( const CBlobDesc& source,
		int filterLen, int strideLen, const CBlobDesc& result ) = 0;
	virtual void BlobMaxOverTimePooling( const CMaxOverTimePoolingDesc& desc, const CFloatHandle& source,
		const CIntHandle* maxIndices, const CFloatHandle& result) = 0;
	virtual void BlobMaxOverTimePoolingBackward( const CMaxOverTimePoolingDesc& desc, const CFloatHandle& outputDiff,
		const CIntHandle& maxIndices, const CFloatHandle& inputDiff ) = 0;

	// Mean-Pooling
	// The descriptor should be destroyed using the standard delete operator after use.
	virtual CMeanPoolingDesc* InitMeanPooling( const CBlobDesc& source,
		int filterHeight, int filterWidth, int strideHeight, int strideWidth,
		const CBlobDesc& result ) = 0;

	virtual void BlobMeanPooling( const CMeanPoolingDesc& desc, const CFloatHandle& source, const CFloatHandle& result ) = 0;
	virtual void BlobMeanPoolingBackward( const CMeanPoolingDesc& desc, const CFloatHandle& outputDiff, const CFloatHandle& inputDiff ) = 0;

	// GlobalMaxOverTime-Pooling
	// The descriptor should be destroyed using the standard delete operator after use.
	virtual CGlobalMaxOverTimePoolingDesc* InitGlobalMaxOverTimePooling( const CBlobDesc& source, const CBlobDesc& result ) = 0;

	virtual void BlobGlobalMaxOverTimePooling( const CGlobalMaxOverTimePoolingDesc& desc, const CFloatHandle& source, const CIntHandle* maxIndices,
		const CFloatHandle& result ) = 0;
	virtual void BlobGlobalMaxOverTimePoolingBackward( const CGlobalMaxOverTimePoolingDesc& desc, const CFloatHandle& source, const CIntHandle& maxIndices,
		const CFloatHandle& result ) = 0;

	virtual void Upsampling2DForward( const CBlobDesc& input, const CConstIntHandle& inputData, int heightCopyCount,
		int widthCopyCount, const CBlobDesc& result, const CIntHandle& resultData ) = 0;
	virtual void Upsampling2DForward( const CBlobDesc& input, const CConstFloatHandle& inputData, int heightCopyCount,
		int widthCopyCount, const CBlobDesc& result, const CFloatHandle& resultData ) = 0;
	virtual void Upsampling2DBackward( const CBlobDesc& input, const CConstFloatHandle& inputData, int heightCopyCount,
		int widthCopyCount, const CBlobDesc& result, const CFloatHandle& resultData ) = 0;

	// Builds a histogram of the number of occurrences in numbersHandle for each integer in [0; maxNumber)
	virtual void BuildIntegerHist( const CConstIntHandle& numbersHandle, int numbersCount,
		const CIntHandle& resultHandle, int maxNumber ) = 0;

	// Calculates the distance for clustering
	virtual void MatrixRowsToVectorSquaredL2Distance( const CConstFloatHandle& matrixHandle, const int matrixHeight,
		const int matrixWidth, const CConstFloatHandle& vectorHandle, const CFloatHandle& resultHandle ) = 0;

	// RLE convolution (works with the specific RLE format of black-and-white 2D images)
	// See CRleConvLayer for more info on RLE format
	// The descriptor should be destroyed using the standard delete operator after use.
	virtual CRleConvolutionDesc* InitBlobRleConvolution( const CBlobDesc& input, float strokeValue,
		float nonStrokeValue, int strideHeight, int strideWidth, const CBlobDesc& filter,
		const CBlobDesc& output ) = 0;

	virtual void BlobRleConvolution( const CRleConvolutionDesc& desc, const CConstFloatHandle& source,
		const CConstFloatHandle& filter, const CConstFloatHandle* freeTerm, const CFloatHandle& result ) = 0;
	virtual void BlobRleConvolutionLearnAdd( const CRleConvolutionDesc& desc, const CConstFloatHandle& input,
		const CConstFloatHandle& outputDiff, const CFloatHandle& filterDiff, const CFloatHandle* freeTermDiff ) = 0;

	// Renumbers the blob elements for a reorg transformation
	// On forward pass, the blob height and width will be reduced by "stride" times, 
	// while the number of channels will be increased stride * stride times
	// An example for stride = 2
	// this channel:
	// 0  1  2  3
	// 4  5  6  7
	// 8  9  10 11
	// 12 13 14 15
	// is transformed into 4 channels:
	// (1) 0 2   (2) 8  10   (3) 1  3   (4) 9  11
	//     4 6       12 14       5  7      13  15
	// If the input has more than one channel, each of them is processed in turn: 
	// the first smaller channel from each, than the second smaller channel from each, etc.
	// On backward pass, the reverse operation is performed: smaller channels are collected together into a larger one
	virtual void Reorg( const CBlobDesc& source, const CFloatHandle& sourceData, int stride, bool isForward,
		const CBlobDesc& result, const CFloatHandle& resultData ) = 0;
	virtual void Reorg( const CBlobDesc& source, const CIntHandle& sourceData, int stride, bool isForward,
		const CBlobDesc& result, const CIntHandle& resultData ) = 0;

	// Rearranges the blob elements from N x H x W x C to N x H / blockSize x W / blockSize x C * blockSize * blockSize
	// The name is chosen to be similar with other frameworks
	virtual void SpaceToDepth( const CBlobDesc& source, const CConstFloatHandle& sourceData, int blockSize,
		const CBlobDesc& result, const CFloatHandle& resultData ) = 0;
	virtual void SpaceToDepth( const CBlobDesc& source, const CConstIntHandle& sourceData, int blockSize,
		const CBlobDesc& result, const CIntHandle& resultData ) = 0;

	// Rearranges the blob elements from N x H x W x C to N x H * blockSize x W * blockSize x C / (blockSize * blockSize)
	// The name is chosen to be similar with other frameworks
	virtual void DepthToSpace( const CBlobDesc& source, const CConstFloatHandle& sourceData, int blockSize,
		const CBlobDesc& result, const CFloatHandle& resultData ) = 0;
	virtual void DepthToSpace( const CBlobDesc& source, const CConstIntHandle& sourceData, int blockSize,
		const CBlobDesc& result, const CIntHandle& resultData ) = 0;

	// To each element, adds its column number (on forward pass)
	// 0 1 2   --->    0 2 4
	// 3 4 5           3 5 7
	// 6 7 8           6 8 10
	// On backward pass, substracts the column number
	virtual void AddWidthIndex( const CBlobDesc& source, const CFloatHandle& sourceData, bool isForward, const CFloatHandle& result ) = 0;
	virtual void AddWidthIndex( const CBlobDesc& source, const CIntHandle& sourceData, bool isForward, const CIntHandle& result ) = 0;

	// To each element, adds its row number (on forward pass)
	// 0 1 2   --->    0 1 2
	// 3 4 5           4 5 6
	// 6 7 8           8 9 10 
	// On backward pass, substracts the row number
	virtual void AddHeightIndex( const CBlobDesc& source, const CFloatHandle& sourceData, bool isForward, const CFloatHandle& result ) = 0;
	virtual void AddHeightIndex( const CBlobDesc& source, const CIntHandle& sourceData, bool isForward, const CIntHandle& result ) = 0;

	// Initializes the dropout descriptor.
	// The dropout descriptor should be destroyed using the standard delete operator after use.
	virtual CDropoutDesc* InitDropout( float rate, bool isSpatial, bool isBatchwise, const CBlobDesc& input,
		const CBlobDesc& output, int seed ) = 0;
	// Performs dropout on an input
	virtual void Dropout( const CDropoutDesc& desc, const CFloatHandle& input, const CFloatHandle& output ) = 0;

	// QRNN poolings (https://arxiv.org/pdf/1611.01576.pdf)
	// These operations calculate recursive parts of QRNN
	// It doesn't include output gate handling
	// Initial states are optional (maybe null)

	// QRNN f-pooling
	virtual void QrnnFPooling( bool reverse, int sequenceLength, int objectSize,
		const CConstFloatHandle& update, const CConstFloatHandle& forget, const CConstFloatHandle& initialState,
		const CFloatHandle& result ) = 0;
	// QRNN f-pooling backward
	// Note: it will modify the contents of resultDiff
	virtual void QrnnFPoolingBackward( bool reverse, int sequenceLength, int objectSize,
		const CConstFloatHandle& update, const CConstFloatHandle& forget,
		const CConstFloatHandle& initialState, const CConstFloatHandle& result, const CFloatHandle& resultDiff,
		const CFloatHandle& updateDiff, const CFloatHandle& forgetDiff ) = 0;

	// QRNN if-pooling
	virtual void QrnnIfPooling( bool reverse, int sequenceLength, int objectSize,
		const CConstFloatHandle& update, const CConstFloatHandle& forget, const CConstFloatHandle& input,
		const CConstFloatHandle& initialState, const CFloatHandle& result ) = 0;
	// QRNN if-pooling backward
	// Note: it will modify the contents of resultDiff
	virtual void QrnnIfPoolingBackward( bool reverse, int sequenceLength, int objectSize,
		const CConstFloatHandle& update, const CConstFloatHandle& forget, const CConstFloatHandle& input,
		const CConstFloatHandle& initialState, const CConstFloatHandle& result, const CFloatHandle& resultDiff,
		const CFloatHandle& updateDiff, const CFloatHandle& forgetDiff, const CFloatHandle& inputDiff ) = 0;

	// Ind-RNN implementation (https://arxiv.org/pdf/1803.04831.pdf)
	// Pay attention that functions below emulate only recurrent part of the layer
	// the result is
	//    h_t = activation( wx + u * h_(t-1))
	// where
	//    wx - user input (x), processed by fully connected layer (w). Size: seqLen x batchSize x objSize
	//    mask - (optional, may be null) dropout mask. Size: batchSize x objSize
	//    u - trainable vector of multipliers. Size: objSize

	// Supported activations
	// - sigmoid
	// - ReLU

	// Inference
	// Calculates h based on wx, mask and u
	virtual void IndRnnRecurrent( bool reverse, int sequenceLength, int batchSize, int objectSize, TActivationFunction activation,
		const CConstFloatHandle& wx, const CConstFloatHandle& mask, const CConstFloatHandle& u, const CFloatHandle& h ) = 0;
	// Backward
	// Calculates wxDiff based on mask, u, h and hDiff
	virtual void IndRnnRecurrentBackward( bool reverse, int sequenceLength, int batchSize, int objectSize,
		TActivationFunction activation, const CConstFloatHandle& mask, const CConstFloatHandle& u, const CConstFloatHandle& h,
		const CConstFloatHandle& hDiff, const CFloatHandle& wxDiff ) = 0;
	// Learn
	// Calculates uDiff based on wx, mask, u, h, and hDiff
	virtual void IndRnnRecurrentLearn( bool reverse, int sequenceLength, int batchSize, int objectSize,
		TActivationFunction activation, const CConstFloatHandle& mask, const CConstFloatHandle& u, const CConstFloatHandle& h,
		const CConstFloatHandle& hDiff, const CFloatHandle& uDiff ) = 0;

	// Local responce normalization (Lrn)
	// For more details see CLrnLayer comments
	virtual CLrnDesc* InitLrn( const CBlobDesc& source, int windowSize, float bias, float alpha, float beta ) = 0;
	// invSum and invSumBeta are required only for backward
	// If you're not gonna use backward, you may pass CFloatHandle()
	// If you need backward, invSum and invSumBeta must be of the same size as input
	virtual void Lrn( const CLrnDesc& desc, const CConstFloatHandle& input, const CFloatHandle& invSum,
		const CFloatHandle& invSumBeta, const CFloatHandle& outputHandle ) = 0;
	virtual void LrnBackward( const CLrnDesc& desc, const CConstFloatHandle& input, const CConstFloatHandle& output,
		const CConstFloatHandle& outputDiff, const CConstFloatHandle& invSum, const CConstFloatHandle& invSumBeta,
		const CFloatHandle& inputDiff ) = 0;

	// If currentDesc isn't nullptr, it will be reinitialized with new values and pointer to it will be returned.
	// Otherwise new descriptor will be created.
	virtual CLstmDesc* InitLstm( CLstmDesc* currentDesc, const CFloatHandle& inputFullyConnectedResult, const CFloatHandle& reccurentFullyConnectedResult,
		int hiddenSize, int objectCount, int objectSize ) = 0;
	virtual void Lstm( CLstmDesc& desc, 
		const CFloatHandle& inputWeights, const CConstFloatHandle& inputFreeTerm,
		const CFloatHandle& recurrentWeights, const CConstFloatHandle& recurrentFreeTerm,
		const CConstFloatHandle& inputStateBackLink, const CConstFloatHandle& inputMainBackLink, const CConstFloatHandle& input,
		const CFloatHandle& outputStateBackLink, const CFloatHandle& outputMainBackLink ) = 0;

	// CTC

	// Calculates CTC loss (and gradient if needed)
	// optional handles may be passed as empty CTypedMemoryHandle<T>()
	// params:
	//     resultLen - number of timesteps in the network result
	//     batchSize - number of independent sequences in the batch
	//     classCount - number of classes in the task (length of the probability vectors)
	//     labelLen - number fo timesteps in the labels
	//     blankLabel - index of the class used for blanks
	//     skipBlank - indicates if blank labels may be skipped during alignment
	// input data:
	//     result - network result (resultLen x batchSize x classCount)
	//     labels - class labels (labelLen x batchSize)
	//     labelLens - (optional) lengths of sequences in labels (batchSize)
	//     resultLens - (optional) lengths of sequences in result (batchSize)
	//     labelWeights - (optional) weights of sequences in the batch (batchSize)
	// output data:
	//     loss - loss value (1)
	//     lossGradient - (optional) lossGradient (resultLen x batchSize x classCount)
	virtual void CtcLossForward(int resultLen, int batchSize, int classCount, int labelLen, int blankLabel, bool skipBlanks,
		const CConstFloatHandle& result, const CConstIntHandle& labels,
		const CConstIntHandle& labelLens, const CConstIntHandle& resultLens, const CConstFloatHandle& labelWeights,
		const CFloatHandle& loss, const CFloatHandle& lossGradient ) = 0;

	// BERT Conv operations
	virtual void BertConv( const CConstFloatHandle& dataHandle, const CConstFloatHandle& kernelHandle,
		int seqLen, int batchSize, int numHeads, int headSize, int kernelSize, const CFloatHandle& outputHandle ) = 0;
	virtual void BertConvBackward( const CConstFloatHandle& dataHandle, const CConstFloatHandle& kernelHandle,
		const CConstFloatHandle& outDiffHandle, int seqLen, int batchSize, int numHeads, int headSize, int kernelSize,
		const CFloatHandle& dataDiffHandle, const CFloatHandle& kernelDiffHandle ) = 0;

	// Linear interpolation

	// data is a 3D tensor of size objectCount x scaledAxis x objectSize
	// result is a 3D tensor of size objectCount x int(scaledAxis * scale) x objectSize
	virtual void LinearInterpolation( const CConstFloatHandle& dataHandle, const CFloatHandle& resultHandle,
		TInterpolationCoords coords, TInterpolationRound round, int objectCount, int scaledAxis, int objectSize, float scale ) = 0;

	// ScatterND
	// Replaces objects in the provided indices with the values from the updates
	// data[indices[i]] = updates[i] for i in [0; updateCount)
	// where indices[i] means coordinates in the first indexDims dimensions of the blob
	// Unaffected data objects remain as-is
	//
	// Data consists of objectCount x objectSize elements where objectCount is a production of first indexDims dimensions
	// Indices consists of updateCount x indexDims elements
	// Updates consists of updateCount x objectCount elements
	virtual void ScatterND( const CConstIntHandle& indicesHandle, const CConstFloatHandle& updatesHandle,
		const CFloatHandle& dataHandle, const CBlobDesc& dataDesc, int updateCount, int indexDims ) = 0;
	virtual void ScatterND( const CConstIntHandle& indicesHandle, const CConstIntHandle& updatesHandle,
		const CIntHandle& dataHandle, const CBlobDesc& dataDesc, int updateCount, int indexDims ) = 0;
};

//------------------------------------------------------------------------------------------------------------

// The position in distributed system
struct CMathEngineDistributedInfo {
	int Thread; // number among all threads
	int Threads; // number of all threads

	CMathEngineDistributedInfo() : Thread( 0 ), Threads( 1 ) {};
	CMathEngineDistributedInfo( int thread, int threads ) : Thread( thread ), Threads( threads ) {};
};

//------------------------------------------------------------------------------------------------------------

// The maximum number of blobs in split or merge operations
const int MaxBlobDescs = 32;

// The type of processing unit used by the math engine
enum TMathEngineType {
	MET_Undefined = 0,
	MET_Cpu,
	MET_Cuda,
	MET_Metal,
	MET_Vulkan,
	MET_Count
};

// Contains the information about a graphics processing unit
struct CMathEngineInfo {
	TMathEngineType Type; // device type
	char Name[256]; // device name
	size_t AvailableMemory; // available memory on device
	int Id; // unique identifier for the device

	CMathEngineInfo() : Type( MET_Undefined ), AvailableMemory( 0 ), Id( 0 ) { Name[0] = 0; }
	CMathEngineInfo( TMathEngineType type, size_t availableMemory, int id ) : Type( type ), AvailableMemory( availableMemory ), Id( id ) { Name[0] = 0; }
};

// CMathEngine class implements an engine to perform calculations on data specified by CMemoryHandle (CFloatHandle)
class NEOMATHENGINE_API IMathEngine : public IDnnEngine {
public:
	virtual ~IMathEngine();
	// Gets the device type
	virtual TMathEngineType GetType() const = 0;
	
	// Gets the device information
	virtual void GetMathEngineInfo( CMathEngineInfo& info ) const = 0;

	// Memory management
	// Turns on and off the memory reuse mode
	// In this mode, the allocated memory blocks will not be deleted on HeapFree() and may be used until CleanUp()
	virtual void SetReuseMemoryMode( bool enable ) = 0;
	virtual CMemoryHandle HeapAlloc( size_t count ) = 0;
	virtual void HeapFree( const CMemoryHandle& handle ) = 0;

	// Allocates typed memory
	template<class T>
	CTypedMemoryHandle<T> HeapAllocTyped( size_t count ) { return CTypedMemoryHandle<T>( HeapAlloc( count * sizeof( T ) ) ); }

	// Stack memory management
	// Alloc and Free functions should be called strictly in LIFO order
	// More efficient than a standard Alloc, recommended for temporary buffers
	// Different stacks are used for different threads
	virtual CMemoryHandle StackAlloc( size_t count ) = 0;
	virtual void StackFree( const CMemoryHandle& handle ) = 0;

	// The current size of free memory on the device
	virtual size_t GetFreeMemorySize() const = 0;

	// Gets the peak memory usage achieved during processing
	virtual size_t GetPeakMemoryUsage() const = 0;

	// The current size of memory in the pools
	virtual size_t GetMemoryInPools() const = 0;

	// Releases all temporary resources allocated for the current thread
	virtual void CleanUp() = 0;

	// Gets a pointer to access the handle memory
	// GetBuffer and ReleaseBuffer should be called strictly in LIFO order
	virtual void* GetBuffer( const CMemoryHandle& handle, size_t pos, size_t size, bool exchange ) = 0;
	virtual void ReleaseBuffer( const CMemoryHandle& handle, void* ptr, bool exchange ) = 0;

	// Data exchange device <-> host.
	virtual void DataExchangeRaw( const CMemoryHandle& result, const void* source, size_t size ) = 0;
	virtual void DataExchangeRaw( void* result, const CMemoryHandle& source, size_t size ) = 0;

	// Typed data exchange
	template<class T>
	void DataExchangeTyped( const CTypedMemoryHandle<T>& result, const T* source, size_t size ) { DataExchangeRaw( result, source, size * sizeof(T) ); }
	template<class T>
	void DataExchangeTyped( T* result, const CTypedMemoryHandle<const T>& source, size_t size ) { DataExchangeRaw( result, source, size * sizeof(T) ); }

	// Creates a handle with data from another math engine
	virtual CMemoryHandle CopyFrom( const CMemoryHandle& handle, size_t size ) = 0;

	// Creates a object for aggregating statistics.
	// This object should be destroyed using the standard delete operator after use.
	virtual IPerformanceCounters* CreatePerformanceCounters() const = 0;

	virtual CMathEngineDistributedInfo GetDistributedInfo() { return CMathEngineDistributedInfo(); }
	virtual void AllReduce( const CFloatHandle& handle, int size ) = 0;
	virtual void Broadcast( const CFloatHandle& handle, int size, int root ) = 0;
	virtual void AbortDistributed() {};
	virtual bool IsDistributed() { return false; }
};

//------------------------------------------------------------------------------------------------------------

// Creates a math engine that uses a CPU for calculations
// You should call SetMathEngineExceptionHandler() before this call
// threadCount is the number of threads that may be used;
// the default value is 0, which means as many threads as the CPU has cores
// This math engine should be destroyed using the standard delete operator after use
NEOMATHENGINE_API IMathEngine* CreateCpuMathEngine( int threadCount, size_t memoryLimit );

// Destroys all global data that is shared between CPU math engines
// Should be called only if there are no running CpuMathEngine instances
NEOMATHENGINE_API void CpuMathEngineCleanUp();

// Gpu math engine flags

// Use tensor cores in cublas (if possible)
// If GPU supports tensor cores this flag leads to faster but less precise GEMM operations
// Works only if MET_Cuda and device is Titan V, geforce 16** or newer

// Half tensor core math - less accurate, supported by all cards with tensor cores
// Incompatible with float math flag below
const int GpuMathEngineCublasUseTensorCoresHalfFlag = 0x1;
// TF32 tensor core math - more accurate, supported by modern cards based on the Ampere architecture
// Incompatible with half math flag above
const int GpuMathEngineCublasUseTensorCoresTF32Flag = 0x2;

// Creates a math engine that uses the recommended GPU for calculations
// Returns null if no GPUs are available
// You should call SetMathEngineExceptionHandler() before this call
// memoryLimit is the limit to memory used for processing (set to 0 to have no limitation); 
// if the limit is exceeded IMathEngineExceptionHandler::OnMemoryError() will be called
// This math engine should be destroyed using the standard delete operator after use
NEOMATHENGINE_API IMathEngine* CreateGpuMathEngine( size_t memoryLimit, int flags = 0 );

// The GPU manager interface
// Allows you to access the information about all available GPUs
class NEOMATHENGINE_API IGpuMathEngineManager : public CCrtAllocatedObject {
public:
	virtual ~IGpuMathEngineManager();
	// Gets the number of available GPUs
	virtual int GetMathEngineCount() const = 0;

	// Gets the information about the GPU by index (may be from 0 to GetMathEngineCount() - 1).
	virtual void GetMathEngineInfo( int index, CMathEngineInfo& info ) const = 0;

	// Creates a math engine on the selected GPU
	// index is the number of the GPU in the list of all available devices (may be from 0 to GetMathEngineCount() - 1)
	// memoryLimit is the limit to memory used for processing (set to 0 to have no limitation); 
	// if the limit is exceeded IMathEngineExceptionHandler::OnMemoryError() will be called
	virtual IMathEngine* CreateMathEngine( int index, size_t memoryLimit, int flags = 0 ) const = 0;
};

// Creates a GPU manager
// Should be destroyed after use with the standard delete operator
// You should call SetMathEngineExceptionHandler() before this call
NEOMATHENGINE_API IGpuMathEngineManager* CreateGpuMathEngineManager();

// Creates `count` cpu MathEngines connected via distributed communicator object
NEOMATHENGINE_API void CreateDistributedCpuMathEngines( IMathEngine** mathEngines, int count );
// Creates `count` gpu MathEngines connected via distributed communicator object
// i-th MathEngine placed on gpu with number devs[i]
NEOMATHENGINE_API void CreateDistributedCudaMathEngines( IMathEngine** mathEngines, int devsCount, const int* cudaDevs );

// Deinitialization function for NeoMathEngine library
// It's recommended to call this function during the unload of the dynamic library
// (esp. if you want to load/unload NeoMathEngine multiple times)
NEOMATHENGINE_API void DeinitializeNeoMathEngine();

} // namespace NeoML

#include <NeoMathEngine/MemoryHandle.inl>
