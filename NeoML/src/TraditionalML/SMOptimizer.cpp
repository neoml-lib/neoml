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
	
	int MatrixSize() const { return matrixSize; }
	// Returns true if the data block has been filled
	bool GetColumn(int i, float*& data);
	// Returns the pointer to the column; may be null
	float* GetColumn(int i) const  { return columns[i].Data; }
	void SwapIndex( int i, int j );

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

void CKernelCache::SwapIndex( int i, int j )
{
    if( i == j ) {
		return;
	}

	CList& columnI = columns[i];
	CList& columnJ = columns[j];
    if( columnI.Next != 0 ) {
		lruDelete( &columnI );
		lruInsert( &columnI );
	}
    if( columnI.Next != 0 ) {
		lruDelete( &columnJ );
		lruInsert( &columnJ );
	}
    swap( columnI.Data, columnJ.Data );

    for( CList* column = lruHead.Next; column != &lruHead; column = column->Next ) {
		swap( column->Data[i], column->Data[j] );
    }
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
	const float* GetBinaryClasses() const { return classes.GetPtr(); }
	// swaps the data on i and j indices
	void SwapIndex( int i, int j );

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

void CKernelMatrix::SwapIndex( int i, int j )
{
	cache.SwapIndex( i, j );
	
	CSparseFloatVectorDesc descI;
	CSparseFloatVectorDesc descJ;
	matrix.GetRow( i, descI );
	matrix.GetRow( j, descJ );
	swap( descI.Indexes, descJ.Indexes );
	swap( descI.Values, descJ.Values );

	swap( classes[i], classes[j] );
	swap( diagonal[i], diagonal[j] );
}


//---------------------------------------------------------------------------------------------------

CSMOptimizer::CSMOptimizer(const CSvmKernel& kernel, const IProblem& _data,
		int _maxIter, double _errorWeight, double _tolerance, bool _shrinking, int cacheSize) :
	data( &_data ),
	maxIter( _maxIter ),
	errorWeight( _errorWeight ),
	tolerance( _tolerance ),
	shrinking( _shrinking ),
	Q( FINE_DEBUG_NEW CKernelMatrix( _data, kernel, cacheSize ) ),
	log( nullptr ),
	l( data->GetVectorCount() ),
	y( Q->GetBinaryClasses() ),
	QD( Q->GetDiagonal() )
{
	weightsMultErrorWeight.SetBufferSize( l );
	for( int i = 0; i < l; ++i ) {
		weightsMultErrorWeight.Add( data->GetVectorWeight( i ) * errorWeight );
	}
	C = weightsMultErrorWeight.GetPtr();
}

CSMOptimizer::~CSMOptimizer() 
{ 
	delete Q;
}

void CSMOptimizer::Optimize( CArray<double>& _alpha, float& freeTerm )
{
	gradient.Empty();
	gradient.Add( -1., l ); // the target function gradient
	g = gradient.GetPtr();
	gradient0.Empty();
	gradient0.Add( 0., l ); // gradient, if we treat free variables as 0
	g0 = gradient0.GetPtr();
	alphaStatusArray.Empty();
	alphaStatusArray.Add( AS_LowerBound, l );
	alphaStatus = alphaStatusArray.GetPtr();

	if( shrinking ) {
		// shrinking does some permutations in coefficients so use internal alpha array
		alphaArray.Empty();
		alphaArray.Add( 0., l ); // the support vectors coefficients
		alpha = alphaArray.GetPtr();
		activeSetArray.SetSize( l );
		activeSet = activeSetArray.GetPtr();
		for( int i = 0; i < l; ++i ) {
			activeSet[i] = i;
		}
		unshrink = false;
	} else {
		_alpha.Empty();
		_alpha.Add( 0., l );
		alpha = _alpha.GetPtr();
	}
	activeSize = l;

	int t;
	int counter = min( l, 1000 );
	for( t = 0; t < maxIter; ++t ) {
		if( --counter == 0 ) {
			counter = min( l, 1000 );
			if( shrinking ) {
				shrink();
			}
			// log progress
			if( log != nullptr ) {
				*log << ".";
			}
		}

		int i, j; 
		if( selectWorkingSet( i, j ) ) {
			reconstructGradient();
			activeSize = l;
			if(log != nullptr) {
				*log << "*";
			}
			if( selectWorkingSet( i, j ) ) {
				break;
			} else {
				// shrink on the next iteration
				counter = 1;
			}
		}

		// Find the optimal values for this pair of coefficients
		optimizePair( i, j );
	}
	if(log != nullptr) {
		*log << "\noptimization finished, #iter = " << t << "\n";
	}
	// Calculate the free term
	freeTerm = calculateFreeTerm();

	if( shrinking ) {
		// put back the solution
		_alpha.SetSize( l );
		for( int i = 0; i < l; ++i ) {
			_alpha[activeSet[i]] = alpha[i];
		}
	}
}

// return 1 if already optimal, return 0 otherwise
// i: maximizes -y_i * grad(f)_i, i in I_up(\alpha)
// j: minimizes the decrease of obj value
//  (if quadratic coefficient <= 0, replace it with tau)
//  -y_j*grad(f)_j < -y_i*grad(f)_i, j in I_low(\alpha)
bool CSMOptimizer::selectWorkingSet( int& outI, int& outJ ) const
{
	double gMax = -Inf;
	double gMax2 = -Inf;
	int gMaxIdx = -1;
	int gMinIdx = -1;
	double objDiffMin = Inf;

	for( int t = 0; t < activeSize; ++t) {
		if( y[t] == 1 ) {
			if( alphaStatus[t] != AS_UpperBound ) {
				if( -g[t] >= gMax ) {
					gMax = -g[t];
					gMaxIdx = t;
				}
			}
		} else if( alphaStatus[t] != AS_LowerBound ) {
			if(g[t] >= gMax) {
				gMax = g[t];
				gMaxIdx = t;
			}
		}
	}

	void ( *updateMinParams )( double, double, const float*, const double*, double, int, double, double,
		int&, double& );
	const float* Q_i = NULL;
	double y_i = 0, QD_i = 0;
	if( gMaxIdx != -1 ) {
		Q_i = Q->GetColumn( gMaxIdx );
		y_i = y[gMaxIdx];
		QD_i = QD[gMaxIdx];

		updateMinParams = []( double gradDiff, double mult, const float* Q_i, const double* QD, double QD_i,
				int j, double y_i, double Tau, int& gMinIdx, double& objDiffMin ) {
			if( gradDiff > 0) {
				double quadCoef = QD_i + QD[j] + mult * y_i * Q_i[j];
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
	} else {
		updateMinParams = []( double, double, const float*, const double*, double, int, double, double,
			int&, double& ) {};
	}

	for( int j = 0; j < activeSize; ++j ) {
		if( y[j] == 1 ) {
			if( alphaStatus[j] != AS_LowerBound ) {
				updateMinParams( gMax + g[j], -2, Q_i, QD, QD_i, j, y_i, Tau, gMinIdx, objDiffMin );
				if( g[j] >= gMax2 ) {
					gMax2 = g[j];
				}
			}
		} else if( alphaStatus[j] != AS_UpperBound ) {
			updateMinParams( gMax - g[j], 2, Q_i, QD, QD_i, j, y_i, Tau, gMinIdx, objDiffMin );
			if( -g[j] >= gMax2 ) {
				gMax2 = -g[j];
			}
		}
	}

	if( gMax + gMax2 < tolerance || gMinIdx == -1 ) {
		return true;
	}

	outI = gMaxIdx;
	outJ = gMinIdx;
	return false;
}

// Optimizes the target function by changing the alpha_i and alpha_j coefficient
// The optimal values are calculated analytically
void CSMOptimizer::optimizePair( int i, int j )
{	
	const float* Q_i = Q->GetColumn(i);
	const float* Q_j = Q->GetColumn(j);
	double C_i = C[i];
	double C_j = C[j];
	double oldAlpha_i = alpha[i];
	double oldAlpha_j = alpha[j];

	if( y[i] != y[j] ) {
		double quadCoef = QD[i] + QD[j] + 2 * Q_i[j];
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
		if(diff > C_i - C_j) {
			if(alpha[i] > C_i) {
				alpha[i] = C_i;
				alpha[j] = C_i - diff;
			}
		} else {
			if(alpha[j] > C_j) {
				alpha[j] = C_j;
				alpha[i] = C_j + diff;
			}
		}
	} else {
		double quadCoef = QD[i] + QD[j] - 2 * Q_i[j];
		if( quadCoef <= 0 )
			quadCoef = Tau;
		double delta = (g[i] - g[j]) / quadCoef;
		double sum = alpha[i] + alpha[j];
		alpha[i] -= delta;
		alpha[j] += delta;

		if(sum > C_i) {
			if(alpha[i] > C_i) {
				alpha[i] = C_i;
				alpha[j] = sum - C_i;
			}
		} else {
			if(alpha[j] < 0) {
				alpha[j] = 0;
				alpha[i] = sum;
			}
		}
		if(sum > C_j) {
			if(alpha[j] > C_j)	{
				alpha[j] = C_j;
				alpha[i] = sum - C_j;
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
		g[k] += Q_i[k] * deltaAlpha_i + Q_j[k] * deltaAlpha_j;
	}

	// Update alphaStatus and g0
	auto updateAlphaStatusAndGrad0 = [&]( int idx, double C_idx ) {
		bool ub = alphaStatus[idx] == AS_UpperBound;
		updateAlphaStatus( idx );
		if( ub != ( alphaStatus[idx] == AS_UpperBound ) ) {
			const float* Q_idx = Q->GetColumn( idx );
			if( ub ) {
				for( int k=0; k < l; ++k ) {
					g0[k] -= C_idx * Q_idx[k];
				}
			} else {
				for( int k=0; k < l; ++k ) {
					g0[k] += C_idx * Q_idx[k];
				}
			}
		}
	};
	updateAlphaStatusAndGrad0( i, C_i );
	updateAlphaStatusAndGrad0( j, C_j );
}

void CSMOptimizer::swapIndex( int i, int j )
{
	Q->SwapIndex( i, j );
	swap( g[i], g[j] );
	swap( alphaStatus[i], alphaStatus[j] );
	swap( alpha[i], alpha[j] );
	swap( activeSet[i], activeSet[j] );
	swap( g0[i], g0[j] );
	swap( C[i], C[j] );
}


// excludes from active set elements that have reached their upper/lower bound
// gMax1: max { -y_i * grad(f)_i | i in I_up(\alpha) }
// gMax2: max { y_i * grad(f)_i | i in I_low(\alpha) }
void CSMOptimizer::shrink()
{
	int i;
	double gMax1 = -Inf;
	double gMax2 = -Inf;

	// find maximal violating pair first
	for( i=0; i < activeSize; ++i ) {
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

	if( !unshrink && gMax1 + gMax2 <= tolerance * 10 ) {
		unshrink = true;
		reconstructGradient();
		activeSize = l;
		if( log != nullptr ) {
			*log << "*";
		}
	}

	for( i = 0; i < activeSize; ++i ) {
		if( canBeShrunk( i, gMax1, gMax2 ) ) {
			--activeSize;
			while( activeSize > i ) {
				if( !canBeShrunk( activeSize, gMax1, gMax2 ) ) {
					swapIndex( i, activeSize );
					break;
				}
				--activeSize;
			}
		}
	}
}

// reconstruct inactive elements of G from G_bar and free variables
void CSMOptimizer::reconstructGradient()
{
	if( activeSize == l ) {
		return;
	}

	int freeCount = 0;
	for( int j = activeSize; j < l; ++j ) {
		g[j] = g0[j] - 1;
	}

	for( int j = 0; j < activeSize; ++j ) {
		if( alphaStatus[j] == AS_Free ) {
			freeCount++;
		}
	}

	if( log != nullptr && 2 * freeCount < activeSize ) {
		*log << "\nWarning: using Shrinking=false may be faster\n";
	}

	if( freeCount * l > 2 * activeSize * ( l - activeSize ) )
	{
		for( int i = activeSize; i < l; ++i ) {
			auto Q_i = Q->GetColumn( i );
			for( int j = 0; j < activeSize; ++j ) {
				if( alphaStatus[j] == AS_Free ) {
					g[i] += alpha[j] * Q_i[j];
				}
			}
		}
	} else {
		for( int i = 0; i < activeSize; ++i ) {
			if( alphaStatus[i] == AS_Free ) {
				auto Q_i = Q->GetColumn( i );
				for( int j = activeSize; j < l; ++j ) {
					g[j] += alpha[i] * Q_i[j];
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
	for(int i = 0; i < l; i++) {
		const double binaryClass = y[i];
		double yGrad = -binaryClass * g[i];
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

