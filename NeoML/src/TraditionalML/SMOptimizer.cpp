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

#include <SMOptimizer.h>

namespace NeoML {

const double CSMOptimizer::Inf = HUGE_VAL;
const double CSMOptimizer::Tau = 1e-12;

// Kernel cache
//
// matrixSize is the matrix size 
// cacheSize is the maximum possible cache size in bytes

class CKernelCache
{
public:
	CKernelCache(int matrixSize, int cacheSize);
	~CKernelCache();
	
	// request Column[0,len)
	// return some position start where [start,len) need to be filled
	// (if start >= len, nothing needs to be filled)
	int GetColumn( int i, float*& data, int len );
	// Returns the pointer to the column (may be null) and sets the len
	float* GetColumn( int i, int& len ) const;
	// Swaps the data associated with indices
	void SwapIndices( int i, int j );

private:
	int matrixSize; // the maximum data array len
	int freeSpace; // the free space in cache (how many float values can fit in) 
	struct CList {
		CList *Prev, *Next;	// a circular list
		float *Column; // the column data
		int Length; // Column[0,Length) is cached in this entry

		CList() { Prev = Next = nullptr; Column = nullptr; Length = 0; }
		~CList() { delete[] Column; }
	};
	CArray<CList> columns; // the array of matrix columns
	CList* c; // raw pointer to columns
	CList lruHead; // the head of the LRU list
	
	void lruDelete(CList *l);
	void lruInsert(CList *l);
};

inline float* CKernelCache::GetColumn( int i, int& len ) const
{
	len = c[i].Length;
	return c[i].Column;
}

CKernelCache::CKernelCache( int _matrixSize, int cacheSize )
	: matrixSize( _matrixSize )
{
	columns.SetSize(matrixSize);
	c = columns.GetPtr();
	freeSpace = cacheSize / sizeof(float);
	freeSpace -= matrixSize * sizeof(CList) / sizeof(float); // the columns array size
	freeSpace = max(freeSpace, 2 * matrixSize);	// at least two columns should fit into cache
	lruHead.Next = lruHead.Prev = &lruHead;
}

CKernelCache::~CKernelCache()
{
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

int CKernelCache::GetColumn( int i, float*& data, int len )
{
	CList* l = c + i;
	if( l->Length != 0 ) {
		lruDelete( l );
	}

	int rest = len - l->Length;
	if( rest > 0 ) {
		while( freeSpace < rest ) {
			CList* old = lruHead.Next;
			lruDelete( old );
			if( old->Length != 0 ) {
				delete[] old->Column;
				freeSpace += old->Length;
				old->Column = nullptr;
				old->Length = 0;
			}
		}

		// reallocate space
		if( l->Column != nullptr ) {
			CList tmp;
			tmp.Column = FINE_DEBUG_NEW float[len];
			memcpy( tmp.Column, l->Column, l->Length * sizeof( float ) );
			swap( l->Column, tmp.Column );
		} else {
			NeoPresume( l->Length == 0 );
			l->Column = FINE_DEBUG_NEW float[len];
		}
		
		freeSpace -= rest;
		swap( l->Length, len );
	}
	
	lruInsert( l );
	data = l->Column;
	return len;
}

void CKernelCache::SwapIndices( int i, int j )
{
	if( i == j ) {
		return;
	}

	if( c[i].Length ) {
		lruDelete( &c[i] );
	}
	if( c[j].Length ) {
		lruDelete( &c[j] );
	}
	swap( c[i].Column, c[j].Column );
	swap( c[i].Length, c[j].Length );
	if( c[i].Length ) {
		lruInsert( &c[i] );
	}
	if( c[j].Length ) {
		lruInsert( &c[j] );
	}

	if( i > j ) {
		swap( i, j );
	}
	for( CList *l = lruHead.Next; l != &lruHead; l = l->Next ) {
		if( l->Length > i ) {
			if( l->Length > j ) {
				swap( l->Column[i], l->Column[j] );
			} else {
				lruDelete( l );
				delete[] l->Column;
				freeSpace += l->Length;
				l->Column = nullptr;
				l->Length = 0;
			}
		}
	}
}

// The kernel matrix CKernelMatrix(i, j) = K(i, j) * y_i * y_j
class CKernelMatrix {
public:
	CKernelMatrix( const IProblem& data, const CSvmKernel& kernel, int cacheSize );

	// Gets the pointer to a column
	const float* GetColumn( int i, int len ) const;
	// Gets the pointer to the diagonal
	const double* GetDiagonal() const { return d; }
	// Gets y[i] binary class
	const float* GetBinaryClasses() const { return y; }
	// Swaps the data on i and j indices
	void SwapIndices( int i, int j );

private:
	CSvmKernel kernel; // the SVM kernel
	mutable CKernelCache cache; // the columns cache
	CArray<CSparseFloatVectorDesc> matrix; // the problem data
	CSparseFloatVectorDesc* x; // raw pointer to data
	CArray<float> classes; // the vector classes
	float* y; // raw pointer to binary classes
	CArray<double> diagonal; // the matrix diagonal
	double* d; // raw pointer to diagonal
};

CKernelMatrix::CKernelMatrix( const IProblem& data, const CSvmKernel& kernel, int cacheSize ) :
	kernel(kernel), 
	cache( data.GetVectorCount(), cacheSize * (1<<20) )
{
	matrix.SetSize( data.GetVectorCount() );
	x = matrix.GetPtr();
	classes.SetSize( data.GetVectorCount() );
	y = classes.GetPtr();
	diagonal.SetSize( data.GetVectorCount() );
	d = diagonal.GetPtr();
	// Calculate the matrix diagonal and fill the matrix with sparse vector descs
	for( int i = 0; i < diagonal.Size(); i++ ) {
		auto& x_i = x[i];
		y[i] = static_cast<float>( data.GetBinaryClass( i ) );
		data.GetMatrix().GetRow( i, x_i );
		d[i] = kernel.Calculate( x_i, x_i );
	}
}

const float* CKernelMatrix::GetColumn( int i, int len ) const
{
	float* column;
	int start = cache.GetColumn( i, column, len );
	if( start < len ) {
		float y_i = y[i];
		auto x_i = x[i];
		auto calcData = [&]( int j ) {
			// the cache matrix is symmetrical so col[i][j] == col[j][i]
			int jColLen;
			float* jColData = cache.GetColumn( j, jColLen );
			if( jColLen > i ) {
				column[j] = jColData[i];
			} else {
				column[j] = static_cast<float>( y_i * y[j] * kernel.Calculate( x_i, x[j] ) );
			}
		};

		// set diagonal element if it's needed
		if( i >= start && i <= len ) {
			for( int j = start; j < i; ++j ) {
				calcData( j );
			}
			column[i] = static_cast<float>( d[i] );
			for( int j = i+1; j < len; ++j ) {
				calcData( j );
			}
		} else {
			for( int j = start; j < len; ++j ) {
				calcData( j );
			}
		}
	}
	return column;
}

void CKernelMatrix::SwapIndices( int i, int j )
{
	cache.SwapIndices( i, j );
	swap( x[i], x[j] );
	swap( y[i], y[j] );
	swap( d[i], d[j] );
}

//---------------------------------------------------------------------------------------------------

CSMOptimizer::CSMOptimizer(const CSvmKernel& kernel, const IProblem& _data,
		int _maxIter, double _errorWeight, double _tolerance, bool _doShrinking, int cacheSize) :
	data( &_data ),
	maxIter( _maxIter ),
	errorWeight( _errorWeight ),
	tolerance( _tolerance ),
	doShrinking( _doShrinking ),
	kernelMatrix( FINE_DEBUG_NEW CKernelMatrix( _data, kernel, cacheSize ) ),
	log( nullptr ),
	vectorCount( data->GetVectorCount() ),
	y( kernelMatrix->GetBinaryClasses() ),
	matrixDiagonal( kernelMatrix->GetDiagonal() )
{
	weightsMultErrorWeightArray.SetBufferSize( vectorCount );
	for( int i = 0; i < vectorCount; ++i ) {
		weightsMultErrorWeightArray.Add( data->GetVectorWeight( i ) * errorWeight );
	}
	weightsMultErrorWeight = weightsMultErrorWeightArray.GetPtr();
}

CSMOptimizer::~CSMOptimizer() 
{ 
	delete kernelMatrix;
}

void CSMOptimizer::Optimize( CArray<double>& _alpha, float& freeTerm )
{
	gradient.Empty();
	gradient.Add( -1., vectorCount ); // the target function gradient
	g = gradient.GetPtr();
	gradient0.Empty();
	gradient0.Add( 0., vectorCount ); // gradient, if we treat free variables as 0
	g0 = gradient0.GetPtr();
	alphaStatusArray.Empty();
	alphaStatusArray.Add( AS_LowerBound, vectorCount );
	alphaStatus = alphaStatusArray.GetPtr();

	if( doShrinking ) {
		// shrinking does some permutations in coefficients so use internal alpha array
		alphaArray.Empty();
		alphaArray.Add( 0., vectorCount ); // the support vectors coefficients
		alpha = alphaArray.GetPtr();
		activeSetArray.SetSize( vectorCount );
		activeSet = activeSetArray.GetPtr();
		for( int i = 0; i < vectorCount; ++i ) {
			activeSet[i] = i;
		}
		isShrunk = false;
	} else {
		_alpha.Empty();
		_alpha.Add( 0., vectorCount );
		alpha = _alpha.GetPtr();
	}
	activeSize = vectorCount;

	int t;
	int counter = min( vectorCount, 1000 ) + 1;
	for( t = 0; t < maxIter; ++t ) {
		if( --counter == 0 ) {
			counter = min( vectorCount, 1000 );
			if( doShrinking ) {
				shrink();
			}
			// log progress
			if( log != nullptr ) {
				*log << ".";
			}
		}
		int i, j; 
		if( !findMaxViolatingIndices( i, j ) ) {
			reconstructGradient();
			if( log != nullptr ) {
				*log << "*";
			}
			if( !findMaxViolatingIndices( i, j ) ) {
				break;
			} else {
				// shrink on the next iteration
				counter = 1;
			}
		}

		// Find the optimal values for this pair of coefficients
		optimizeIndices( i, j );

		// Update alphaStatus and g0
		updateAlphaStatusAndGradient0( i );
		updateAlphaStatusAndGradient0( j );
	}
	
	// Calculate the free term
	freeTerm = calculateFreeTerm();
	if(log != nullptr) {
		*log << "\noptimization finished, #iter = " << t << "\n";
		*log << "freeTerm = " << freeTerm << "\n";
	}

	if( doShrinking ) {
		// put back the solution
		_alpha.SetSize( vectorCount );
		for( int i = 0; i < vectorCount; ++i ) {
			_alpha[activeSet[i]] = alpha[i];
		}
	}
}

// return `false` if already optimal, return `true` otherwise
// i: maximizes -y_i * grad(f)_i, i in I_up(\alpha)
// j: minimizes the decrease of obj value
//  (if quadratic coefficient <= 0, replace it with tau)
//  -y_j*grad(f)_j < -y_i*grad(f)_i, j in I_low(\alpha)
bool CSMOptimizer::findMaxViolatingIndices( int& outI, int& outJ ) const
{
	double gMax = -Inf;
	double gMax2 = -Inf;
	int gMaxIdx = -1;
	int gMinIdx = -1;
	double objDiffMin = Inf;

	for( int i = 0; i < activeSize; ++i) {
		if( y[i] == 1 ) {
			if( alphaStatus[i] != AS_UpperBound ) {
				if( -g[i] >= gMax ) {
					gMax = -g[i];
					gMaxIdx = i;
				}
			}
		} else if( alphaStatus[i] != AS_LowerBound ) {
			if(g[i] >= gMax) {
				gMax = g[i];
				gMaxIdx = i;
			}
		}
	}

	if( gMaxIdx == -1 ) {
		return false;
	}

	const float* q_i = kernelMatrix->GetColumn( gMaxIdx, activeSize );
	double y_i = y[gMaxIdx];
	double qD_i = matrixDiagonal[gMaxIdx];
	auto updateMinParams = [&]( double gradDiff, double multiplier, int j ) {
		if( gradDiff > 0) {
			double quadCoef = qD_i + matrixDiagonal[j] + multiplier * y_i * q_i[j];
			if( quadCoef <= 0 ) {
				quadCoef = Tau;
			}
			double objDiff = -( gradDiff * gradDiff ) / quadCoef;
			if( objDiff <= objDiffMin ) {
				gMinIdx = j;
				objDiffMin = objDiff;
			}
		}
	};

	for( int j = 0; j < activeSize; ++j ) {
		if( y[j] == 1 ) {
			if( alphaStatus[j] != AS_LowerBound ) {
				updateMinParams( gMax + g[j], -2, j );
				if( g[j] >= gMax2 ) {
					gMax2 = g[j];
				}
			}
		} else if( alphaStatus[j] != AS_UpperBound ) {
			updateMinParams( gMax - g[j], 2, j );
			if( -g[j] >= gMax2 ) {
				gMax2 = -g[j];
			}
		}
	}

	if( gMax + gMax2 < tolerance || gMinIdx == -1 ) {
		return false;
	}

	outI = gMaxIdx;
	outJ = gMinIdx;
	return true;
}

// Optimizes the target function by changing the alpha_i and alpha_j coefficient
// The optimal values are calculated analytically
void CSMOptimizer::optimizeIndices( int i, int j )
{	
	const float* q_i = kernelMatrix->GetColumn( i, activeSize );
	const float* q_j = kernelMatrix->GetColumn( j, activeSize );
	double c_i = weightsMultErrorWeight[i];
	double c_j = weightsMultErrorWeight[j];
	double oldAlpha_i = alpha[i];
	double oldAlpha_j = alpha[j];

	if( y[i] != y[j] ) {
		double quadCoef = matrixDiagonal[i] + matrixDiagonal[j] + 2 * q_i[j];
		if( quadCoef <= 0) {
			quadCoef = Tau;
		}
		double delta = (-g[i] - g[j]) / quadCoef;
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
		if(diff > c_i - c_j) {
			if(alpha[i] > c_i) {
				alpha[i] = c_i;
				alpha[j] = c_i - diff;
			}
		} else {
			if(alpha[j] > c_j) {
				alpha[j] = c_j;
				alpha[i] = c_j + diff;
			}
		}
	} else {
		double quadCoef = matrixDiagonal[i] + matrixDiagonal[j] - 2 * q_i[j];
		if( quadCoef <= 0 )
			quadCoef = Tau;
		double delta = (g[i] - g[j]) / quadCoef;
		double sum = alpha[i] + alpha[j];
		alpha[i] -= delta;
		alpha[j] += delta;

		if(sum > c_i) {
			if(alpha[i] > c_i) {
				alpha[i] = c_i;
				alpha[j] = sum - c_i;
			}
		} else {
			if(alpha[j] < 0) {
				alpha[j] = 0;
				alpha[i] = sum;
			}
		}
		if(sum > c_j) {
			if(alpha[j] > c_j)	{
				alpha[j] = c_j;
				alpha[i] = sum - c_j;
			}
		} else {
			if(alpha[i] < 0) {
				alpha[i] = 0;
				alpha[j] = sum;
			}
		}
	}
	
	// Modify the g
	double deltaAlpha_i = alpha[i] - oldAlpha_i;
	double deltaAlpha_j = alpha[j] - oldAlpha_j;
	for(int k = 0; k < activeSize; k++) {
		g[k] += q_i[k] * deltaAlpha_i + q_j[k] * deltaAlpha_j;
	}
}

void CSMOptimizer::updateAlphaStatusAndGradient0( int i )
{
	double c_i = weightsMultErrorWeight[i];
	bool wasUB = alphaStatus[i] == AS_UpperBound;

	if( alpha[i] >= c_i ) {
		alphaStatus[i] = AS_UpperBound;
	} else if( alpha[i] <= 0 ) {
		alphaStatus[i] = AS_LowerBound;
	} else {
		alphaStatus[i] = AS_Free;
	}

	bool isUB = alphaStatus[i] == AS_UpperBound;
	if( wasUB != isUB ) {
		auto q_i = kernelMatrix->GetColumn( i, vectorCount );
		if( wasUB ) {
			for( int j = 0; j < vectorCount; ++j ) {
				g0[j] -= c_i * q_i[j];
			}
		} else {
			for( int j = 0; j < vectorCount; ++j ) {
				g0[j] += c_i * q_i[j];
			}
		}
	}
}

// reconstruct inactive elements of G from G_bar and free variables
void CSMOptimizer::reconstructGradient()
{
	if( activeSize == vectorCount ) {
		return;
	}

	int freeCount = 0;
	for( int j = activeSize; j < vectorCount; ++j ) {
		g[j] = g0[j] - 1;
	}

	for( int j = 0; j < activeSize; ++j ) {
		if( alphaStatus[j] == AS_Free ) {
			++freeCount;
		}
	}

	if( log != nullptr && 2 * freeCount < activeSize ) {
		NeoPresume( doShrinking );
		*log << "\nWarning: using Shrinking=false may be faster\n";
	}

	if( freeCount * vectorCount > 2 * activeSize * ( vectorCount - activeSize ) ) {
		for( int i = activeSize; i < vectorCount; ++i ) {
			auto q_i = kernelMatrix->GetColumn( i, activeSize );
			for( int j = 0; j < activeSize; ++j ) {
				if( alphaStatus[j] == AS_Free ) {
					g[i] += alpha[j] * q_i[j];
				}
			}
		}
	} else {
		for( int i = 0; i < activeSize; ++i ) {
			if( alphaStatus[i] == AS_Free ) {
				auto q_i = kernelMatrix->GetColumn( i, vectorCount );
				double alpha_i = alpha[i];
				for( int j = activeSize; j < vectorCount; ++j ) {
					g[j] += alpha_i * q_i[j];
				}
			}
		}
	}
	activeSize = vectorCount;
}

void CSMOptimizer::swapIndices( int i, int j )
{
	NeoPresume( doShrinking );

	kernelMatrix->SwapIndices( i, j );
	swap( g[i], g[j] );
	swap( g0[i], g0[j] );
	swap( alpha[i], alpha[j] );
	swap( weightsMultErrorWeight[i], weightsMultErrorWeight[j] );
	swap( alphaStatus[i], alphaStatus[j] );
	swap( activeSet[i], activeSet[j] );
}

// excludes from active set elements that have reached their upper/lower bound
// gMax1: max { -y_i * grad(f)_i | i in I_up(\alpha) }
// gMax2: max { y_i * grad(f)_i | i in I_low(\alpha) }
void CSMOptimizer::shrink()
{
	double gMax1 = -Inf;
	double gMax2 = -Inf;

	// find maximal violating pair first
	for( int i = 0; i < activeSize; ++i ) {
		if( y[i] == 1 ) {
			if( alphaStatus[i] != AS_UpperBound && -g[i] >= gMax1 ) {
				gMax1 = -g[i];
			}
			if( alphaStatus[i] != AS_LowerBound && g[i] >= gMax2 ) {
				gMax2 = g[i];
			}
		} else {
			if( alphaStatus[i] != AS_UpperBound && -g[i] >= gMax2 ) {
				gMax2 = -g[i];
			}
			if( alphaStatus[i] != AS_LowerBound && g[i] >= gMax1 ) {
				gMax1 = g[i];
			}
		}
	}

	if( !isShrunk && gMax1 + gMax2 <= tolerance * 10 ) {
		isShrunk = true;
		reconstructGradient();
		if( log != nullptr ) {
			*log << "*";
		}
	}

	for( int i = 0; i < activeSize; ++i ) {
		if( canBeShrunk( i, gMax1, gMax2 ) ) {
			while( --activeSize > i ) {
				if( !canBeShrunk( activeSize, gMax1, gMax2 ) ) {
					swapIndices( i, activeSize );
					break;
				}
			}
		}
	}
}

// Calculates the free term
float CSMOptimizer::calculateFreeTerm() const
{
	int nFree = 0; // the number of "free" support vectors
	double upperBound = Inf, lowerBound = -Inf, sumFree = 0;
	for(int i = 0; i < activeSize; i++) {
		const double binaryClass = y[i];
		double yGrad = -binaryClass * g[i];
		if( alphaStatus[i] == AS_UpperBound ) {
			if(binaryClass == +1) {
				upperBound = min(upperBound, yGrad);
			} else {
				lowerBound = max(lowerBound, yGrad);
			}
		} else if( alphaStatus[i] == AS_LowerBound ) {
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

