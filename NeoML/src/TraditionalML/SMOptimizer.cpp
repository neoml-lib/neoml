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

// This method is described in
// "Working Set Selection Using Second Order Information for Training Support Vector Machines"
// Rong-En Fan, Pai-Hsuen Chen, Chih-Jen Lin, 
// "Journal of Machine Learning Research" 6 (2005) 1889–1918
// http://www.csie.ntu.edu.tw/~cjlin/papers/quadworkset.pdf

#include <common.h>
#pragma hdrstop

#include <NeoML/TraditionalML/SMOptimizer.h>

namespace NeoML {

const double CSMOptimizer::Inf = HUGE_VAL;
const double CSMOptimizer::Tau = 1e-12;

// Raise a number to a power: base**times
inline double power(double base, int times)
{
	double tmp = base, ret = 1.0;
	for(int t = times; t > 0; t /= 2)
	{
		if(t % 2 == 1) {
			ret *= tmp;
		}
		tmp = tmp * tmp;
	}
	return ret;
}

CSvmKernel::CSvmKernel(TKernelType kernelType, int degree, double gamma, double coef0) :
	kernelType(kernelType), degree(degree), gamma(gamma), coef0(coef0)
{
}

// The linear kernel
double CSvmKernel::linear(const CSparseFloatVectorDesc& x1, const CSparseFloatVectorDesc& x2) const
{
	return DotProduct(x1, x2);
}

double CSvmKernel::linear(const CFloatVector& x1, const CSparseFloatVectorDesc& x2) const
{
	return DotProduct(x1, x2);
}

// The polynomial kernel
double CSvmKernel::poly(const CSparseFloatVectorDesc& x1, const CSparseFloatVectorDesc& x2) const
{
	return power(gamma * DotProduct(x1, x2) + coef0, degree);
}

double CSvmKernel::poly(const CFloatVector& x1, const CSparseFloatVectorDesc& x2) const
{
	return power(gamma * DotProduct(x1, x2) + coef0, degree);
}

// The Gaussian kernel
double CSvmKernel::rbf(const CSparseFloatVectorDesc& x1, const CSparseFloatVectorDesc& x2) const
{
	double square = 0;
	int i, j;
	double diff;
	for( i = 0, j = 0; i < x1.Size && j < x2.Size; ) {
		if( x1.Indexes[i] == x2.Indexes[j] ) {
			diff = x1.Values[i] - x2.Values[j];
			i++;
			j++;
		} else if( x1.Indexes[i] < x2.Indexes[j] ) {
			diff = x1.Values[i];
			i++;
		} else {
			diff = x2.Values[j];
			j++;
		}
		square += diff * diff;
	}
	for(; i < x1.Size; i++) {
		diff = x1.Values[i];
		square += diff * diff;
	}
	for(; j < x2.Size; j++) {
		diff = x2.Values[j];
		square += diff * diff;
	}
	return exp(-gamma * square);
}

double CSvmKernel::rbf(const CFloatVector& x1, const CSparseFloatVectorDesc& x2) const
{
	double square = 0;
	int i, j;
	double diff;
	for( i = 0, j = 0; i < x1.Size() && j < x2.Size; ) {
		if( i == x2.Indexes[j] ) {
			diff = x1[i] - x2.Values[j];
			i++;
			j++;
		} else if( i < x2.Indexes[j] ) {
			diff = x1[i];
			i++;
		} else {
			diff = x2.Values[j];
			j++;
		}
		square += diff * diff;
	}
	for(; i < x1.Size(); i++) {
		diff = x1[i];
		square += diff * diff;
	}
	for(; j < x2.Size; j++) {
		diff = x2.Values[j];
		square += diff * diff;
	}
	return exp(-gamma * square);
}

// The sigmoid kernel
double CSvmKernel::sigmoid(const CSparseFloatVectorDesc& x1, const CSparseFloatVectorDesc& x2) const
{
	return tanh(gamma * DotProduct(x1, x2) + coef0);
}

double CSvmKernel::sigmoid(const CFloatVector& x1, const CSparseFloatVectorDesc& x2) const
{
	return tanh(gamma * DotProduct(x1, x2) + coef0);
}

double CSvmKernel::Calculate(const CSparseFloatVectorDesc& x1, const CSparseFloatVectorDesc& x2) const
{
	switch( kernelType ) {
		case KT_Linear:
			return linear(x1, x2);
		case KT_Poly:
			return poly(x1, x2);
		case KT_RBF:
			return rbf(x1, x2);
		case KT_Sigmoid:
			return sigmoid(x1, x2);
		default:
			NeoAssert(false);
			return 0;
	}
}

double CSvmKernel::Calculate(const CFloatVector& x1, const CSparseFloatVectorDesc& x2) const
{
	switch( kernelType ) {
		case KT_Linear:
			return linear(x1, x2);
		case KT_Poly:
			return poly(x1, x2);
		case KT_RBF:
			return rbf(x1, x2);
		case KT_Sigmoid:
			return sigmoid(x1, x2);
		default:
			NeoAssert(false);
			return 0;
	}
}

// Kernel cache
//
// matrixSize is the matrix size 
// cacheSize is the maximum possible cache size in bytes

class CKernelCache
{
public:
	CKernelCache(int matrixSize, int cacheSize);
	~CKernelCache();
	
	int MatrixSize() const { return matrixSize; }
	// Returns true if the data block has been filled
	bool GetColumn(int i, float*& data);
	// Returns the pointer to the column; may be null
	float* GetColumn(int i) const  { return columns[i].Data; }

private:
	const int matrixSize; // the matrix size
	int freeSpace; // the free space in cache (how many float values can fit in) 
	struct CList {
		CList *Prev, *Next;	// circular list
		float *Data; // the data

		CList() { Prev = Next = 0; Data = 0; }
	};
	CArray<CList> columns;  // the array of matrix columns
	CList lruHead; // the head of the LRU list
	
	void lruDelete(CList *l);
	void lruInsert(CList *l);
};

CKernelCache::CKernelCache(int matrixSize, int cacheSize) : matrixSize(matrixSize)
{
	columns.SetSize(matrixSize);
	freeSpace = cacheSize / sizeof(float);
	freeSpace -= matrixSize * sizeof(CList) / sizeof(float); // the columns array size
	freeSpace = max(freeSpace, 2 * matrixSize);	// at least two columns should fit into cache
	lruHead.Next = lruHead.Prev = &lruHead;
}

CKernelCache::~CKernelCache()
{
	for(CList *l = lruHead.Next; l != &lruHead; l = l->Next) {
		delete l->Data;
	}
}

// Deletes an element from the LRU list
void CKernelCache::lruDelete(CList *l)
{
	l->Prev->Next = l->Next;
	l->Next->Prev = l->Prev;
}

// Inserts an element into the end of the LRU list (so it will be deleted last)
void CKernelCache::lruInsert(CList *l)
{
	l->Next = &lruHead;
	l->Prev = lruHead.Prev;
	l->Prev->Next = l;
	l->Next->Prev = l;
}

bool CKernelCache::GetColumn(int i, float*& data)
{
	CList& column = columns[i];
	if(column.Next != 0) {
		lruDelete(&column);
	}
	lruInsert(&column);
	// The cache has the necessary data
	if(column.Data != 0) {
		data = column.Data;
		return true;
	}
	// Free up space
	if(freeSpace < matrixSize) {
		CList *old = lruHead.Next;
		lruDelete(old);
		delete old->Data;
		old->Data = 0;
		freeSpace += matrixSize;
	}
	// Allocate a buffer for new data
	column.Data = FINE_DEBUG_NEW float[matrixSize];
	freeSpace -= matrixSize;
	data = column.Data;
	return false;
}

// The kernel matrix CKernelMatrix(i, j) = K(i, j) * y_i * y_j
class CKernelMatrix {
public:
	CKernelMatrix( const IProblem& data, const CSvmKernel& kernel, int cacheSize );

	// The size of a square matrix
	int Size() const { return diagonal.Size(); }
	// The SVM kernel
	const CSvmKernel& Kernel() const { return kernel; }
	// Gets the pointer to a column
	const float* GetColumn(int i) const;
	// Gets the pointer to the diagonal
	const double* GetDiagonal() const { return diagonal.GetPtr(); }
	// Gets y[i] binary class
	float GetBinaryClass( int i ) const  { return classes[i]; }

private:
	const CSparseFloatMatrixDesc matrix; // the problem data
	CArray<float> classes; // the vector classes
	CSvmKernel kernel; // the SVM kernel
	mutable CKernelCache cache; // the columns cache
	CArray<double> diagonal; // the matrix diagonal
};

CKernelMatrix::CKernelMatrix( const IProblem& data, const CSvmKernel& kernel, int cacheSize ) :
	matrix( data.GetMatrix() ),
	kernel(kernel), 
	cache( data.GetVectorCount(), cacheSize * (1<<20) )
{
	diagonal.SetSize( data.GetVectorCount() );
	// Calculate the matrix diagonal
	for( int i = 0; i < diagonal.Size(); i++ ) {
		classes.Add( static_cast<float>( data.GetBinaryClass(i) ) );
		CSparseFloatVectorDesc vector;
		matrix.GetRow( i, vector );
		diagonal[i] = kernel.Calculate( vector, vector );
	}
}

const float* CKernelMatrix::GetColumn(int i) const
{
	float* columnData;
	if( !cache.GetColumn(i, columnData) ) {
		// Fill the cache with data
		for( int j = 0; j < Size(); j++ ) {
			if(i == j) {
				columnData[j] = static_cast<float>(diagonal[i]);
			} else {
				float* columnData1 = cache.GetColumn(j);
				if(columnData1 != 0) {
					columnData[j] = columnData1[i]; // the matrix is symmetrical
				} else {
					CSparseFloatVectorDesc descI;
					CSparseFloatVectorDesc descJ;
					matrix.GetRow( i, descI );
					matrix.GetRow( j, descJ );
					columnData[j] = static_cast<float>( classes[i] * classes[j] * kernel.Calculate( descI, descJ ) );
				}
			}
		}
	}
	return columnData;
}

//---------------------------------------------------------------------------------------------------

CSMOptimizer::CSMOptimizer(const CSvmKernel& kernel, const IProblem& _data,
		int _maxIter, double _errorWeight, double tolerance, int cacheSize) :
	data( &_data ),
	maxIter( _maxIter ),
	errorWeight( _errorWeight ),
	tolerance(tolerance),
	Q( FINE_DEBUG_NEW CKernelMatrix( _data, kernel, cacheSize ) ),
	log(0)
{
	C.SetBufferSize( data->GetVectorCount() );
	for( int i = 0; i < data->GetVectorCount(); ++i ) {
		C.Add( data->GetVectorWeight( i ) * errorWeight );
	}
}

CSMOptimizer::~CSMOptimizer() 
{ 
	delete Q;
}

void CSMOptimizer::Optimize( CArray<double>& _alpha, float& freeTerm )
{
	gradient.Empty();
	gradient.Add(-1., data->GetVectorCount() ); // the target function gradient
	_alpha.Empty();
	_alpha.Add( 0., data->GetVectorCount() ); // the support vectors coefficients
	alpha = _alpha.GetPtr();

	int t;
	int counter = min( data->GetVectorCount(), 1000 );
	for(t = 0; t < maxIter; t++) {
		// log progress
		if( --counter == 0 ) {
			counter = min( data->GetVectorCount(), 1000 );
			if( log != 0 ) {
				*log << ".";
			}
		}

		// Find a pair of coefficients that violate Kuhn - Tucker conditions most of all
		int i, j; 
		double Gmax, Gmin;
		findMaximalViolatingPair( alpha, gradient, i, Gmax, j, Gmin);
		if(Gmax - Gmin < tolerance)	{
			break;
		}
		// Find the optimal values for this pair of coefficients
		optimizePair( i, j, alpha, gradient );
	}
	if(log != 0) {
		*log << "\noptimization finished, #iter = " << t << "\n";
	}
	// Calculate the free term
	freeTerm = calculateFreeTerm( alpha, gradient );
}

// reconstruct inactive elements of G from G_bar and free variables
void SMOptimizer::reconstructGradient()
{
	const int l = data->GetVectorCount();
	if( activeSize == l ) {
		return;
	}

	int freeCount = 0;
	for( int j = activeSize; j < l; ++j ) {
		gradient[j] = gradient0[j] - 1;
	}

	for( int j=0; j < activeSize; ++j ) {
		if( alphaStatus[j] == FREE ) {
			freeCount++;
		}
	}

	if( freeCount * l > 2 * activeSize * ( l - activeSize ) )
	{
		for( int i = activeSize; i < l; ++i ) {
			auto Q_i = Q->GetColumn( i );
			for( int j=0; j < activeSize; ++j ) {
				if( alphaStatus[j] == FREE )
					gradient[i] += alpha[j] * Q_i[j];
			}
		}
	} else {
		for( int i = 0; i < activeSize; ++i ) {
			if( alphaStatus[i] == FREE ) {
				auto Q_i = Q->GetColumn( i );
				for( int j = activeSize; j<l; ++j )
					gradient[j] += alpha[i] * Q_i[j];
			}
		}
	}
}

// Optimizes the target function by changing the alpha_i and alpha_j coefficient
// The optimal values are calculated analytically
void CSMOptimizer::optimizePair( int i, int j )
{	
	const float* Q_i = Q->GetColumn(i);
	const float* Q_j = Q->GetColumn(j);
	const double* QD = Q->GetDiagonal();

	double oldAlpha_i = alpha[i];
	double oldAlpha_j = alpha[j];

	if( Q->GetBinaryClass(i) != Q->GetBinaryClass(j) ) {
		double quadCoef = QD[i] + QD[j] + 2 * Q_i[j];
		if (quadCoef <= 0) {
			quadCoef = Tau;
		}
		double delta = (-gradient[i] - gradient[j]) / quadCoef;
		double diff = alpha[i] - alpha[j];
		alpha[i] += delta;
		alpha[j] += delta;
			
		if(diff > 0) {
			if(alpha[j] < 0) {
				alpha[j] = 0;
				alpha[i] = diff;
			}
		} else {
			if(alpha[i] < 0) {
				alpha[i] = 0;
				alpha[j] = -diff;
			}
		}
		if(diff > C[i] - C[j]) {
			if(alpha[i] > C[i]) {
				alpha[i] = C[i];
				alpha[j] = C[i] - diff;
			}
		} else {
			if(alpha[j] > C[j]) {
				alpha[j] = C[j];
				alpha[i] = C[j] + diff;
			}
		}
	} else {
		double quadCoef = QD[i] + QD[j] - 2 * Q_i[j];
		if (quadCoef <= 0)
			quadCoef = Tau;
		double delta = (gradient[i] - gradient[j]) / quadCoef;
		double sum = alpha[i] + alpha[j];
		alpha[i] -= delta;
		alpha[j] += delta;

		if(sum > C[i]) {
			if(alpha[i] > C[i]) {
				alpha[i] = C[i];
				alpha[j] = sum - C[i];
			}
		} else {
			if(alpha[j] < 0) {
				alpha[j] = 0;
				alpha[i] = sum;
			}
		}
		if(sum > C[j]) {
			if(alpha[j] > C[j])	{
				alpha[j] = C[j];
				alpha[i] = sum - C[j];
			}
		} else {
			if(alpha[i] < 0) {
				alpha[i] = 0;
				alpha[j] = sum;
			}
		}
	}
	// Modify the gradient
	double deltaAlpha_i = alpha[i] - oldAlpha_i;
	double deltaAlpha_j = alpha[j] - oldAlpha_j;
	for(int k = 0; k < data->GetVectorCount(); k++) {
		gradient[k] += Q_i[k] * deltaAlpha_i + Q_j[k] * deltaAlpha_j;
	}
}


// Calculates the free term
float CSMOptimizer::calculateFreeTerm( const CArray<double>& gradient ) const
{
	int nFree = 0; // the number of "free" support vectors
	double upperBound = Inf, lowerBound = -Inf, sumFree = 0;
	for(int i = 0; i < data->GetVectorCount(); i++) {
		const double binaryClass = Q->GetBinaryClass(i);
		double yGrad = -binaryClass * gradient[i];
		if(alpha[i] >= C[i]) {
			if(binaryClass == +1) {
				upperBound = min(upperBound, yGrad);
			} else {
				lowerBound = max(lowerBound, yGrad);
			}
		} else if(alpha[i] <= 0) {
			if(binaryClass == -1) {
				upperBound = min(upperBound, yGrad);
			} else {
				lowerBound = max(lowerBound, yGrad);
			}
		} else {
			nFree += 1;
			sumFree += yGrad;
		}
	}

	if(nFree > 0) {
		return static_cast<float>( sumFree / nFree );
	}
	return static_cast<float>( (lowerBound + upperBound) / 2 );
}

} // namespace NeoML

