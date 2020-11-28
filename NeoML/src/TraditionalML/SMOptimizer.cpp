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
	
	// request data [0,len)
	// return some position p where [p,len) need to be filled
	// (p >= len if nothing needs to be filled)
	int GetData( int i, float*& data, int len );
	// Returns the pointer to the column (may be null) and sets the len
	float* GetData( int i, int& len ) const;
	void SwapIndex( int i, int j );

private:
	int freeSpace; // the free space in cache (how many float values can fit in) 
	struct CList {
		CList *Prev, *Next;	// a circular list
		float *Data; // the data
		int Length; // data[0,Length) is cached in this entry

		CList() { Prev = Next = nullptr; Data = nullptr; Length = 0; }
	};
	CArray<CList> columns; // the array of matrix columns
	CList* c; // raw pointer to columns
	CList lruHead; // the head of the LRU list
	
	void lruDelete(CList *l);
	void lruInsert(CList *l);
};

inline float* CKernelCache::GetData( int i, int& len ) const
{
	len = c[i].Length;
	return c[i].Data;
}

CKernelCache::CKernelCache( int matrixSize, int cacheSize )
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

int CKernelCache::GetData( int i, float*& data, int len )
{
    CList *h = &c[i];
    if(h->Length) lruDelete(h);
    int more = len - h->Length;

    if(more > 0)
    {
        // free old space
        while( freeSpace < more )
        {
            CList *old = lruHead.Next;
            lruDelete(old);
            delete old->Data;
            freeSpace += old->Length;
            old->Data = nullptr;
            old->Length = 0;
        }

        // allocate new space
		h->Data = FINE_DEBUG_NEW float[len];
        freeSpace -= more;
        swap( h->Length, len );
    }

    lruInsert(h);
    data = h->Data;
    return len;
}

void CKernelCache::SwapIndex( int i, int j )
{
	if( i == j ) {
		return;
	}

	CList* head = c;
    if(head[i].Length) lruDelete(&head[i]);
    if(head[j].Length) lruDelete(&head[j]);
    swap(head[i].Data,head[j].Data);
    swap(head[i].Length,head[j].Length);
    if(head[i].Length) lruInsert(&head[i]);
    if(head[j].Length) lruInsert(&head[j]);

    if(i>j) swap(i,j);
    for(CList *h = lruHead.Next; h!=&lruHead; h=h->Next)
    {
        if(h->Length > i)
        {
            if(h->Length > j)
                swap(h->Data[i],h->Data[j]);
            else
            {
                // give up
                lruDelete(h);
                delete h->Data;
                freeSpace += h->Length;
                h->Data = 0;
                h->Length = 0;
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
	// swaps the data on i and j indices
	void SwapIndex( int i, int j );

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
		y[i] = data.GetBinaryClass( i );
		data.GetMatrix().GetRow( i, x_i );
		d[i] = kernel.Calculate( x_i, x_i );
	}
}

const float* CKernelMatrix::GetColumn( int i, int len ) const
{
	float *data;
	int start;
	if( ( start = cache.GetData( i, data, len ) ) < len ) {
		float y_i = y[i];
		auto x_i = x[i];
		auto calcData = [&]( int j ) {
			// the cache matrix is symmetrical so col[i][j] == col[j][i]
			int jColLen;
			float* jColData = cache.GetData( j, jColLen );
			if( jColLen > i ) {
				data[j] = jColData[i];
			} else {
				data[j] = static_cast<float>( y_i * y[j] * kernel.Calculate( x_i, x[j] ) );
			}
		};

		if( i >= start && i <= len ) {
			for( int j = start; j < i; ++j ) {
				calcData( j );
			}
			data[i] = d[i];
			for( int j = i+1; j < len; ++j ) {
				calcData( j );
			}
		} else {
			for( int j = start; j < len; ++j ) {
				calcData( j );
			}
		}
	}
	return data;
}

void CKernelMatrix::SwapIndex( int i, int j )
{
	cache.SwapIndex( i, j );
	swap( x[i], x[j] );
	swap( y[i], y[j] );
	swap( d[i], d[j] );
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

		// Update alphaStatus and g0
		updateAlphaStatusAndGradient0( i );
		updateAlphaStatusAndGradient0( j );
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
		Q_i = Q->GetColumn( gMaxIdx, activeSize );
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
	const float* Q_i = Q->GetColumn( i, activeSize );
	const float* Q_j = Q->GetColumn( j, activeSize );
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
}

void CSMOptimizer::updateAlphaStatusAndGradient0( int i )
{
	double C_i = C[i];
	bool wasUB = alphaStatus[i] == AS_UpperBound;

	if( alpha[i] >= C_i ) {
		alphaStatus[i] = AS_UpperBound;
	} else if( alpha[i] <= 0 ) {
		alphaStatus[i] = AS_LowerBound;
	} else {
		alphaStatus[i] = AS_Free;
	}

	bool isUB = alphaStatus[i] == AS_UpperBound;
	if( wasUB != isUB ) {
		auto Q_i = Q->GetColumn( i, l );
		if( wasUB ) {
			for( int k = 0; k < l; ++k ) {
				g0[k] -= C_i * Q_i[k];
			}
		} else {
			for( int k = 0; k < l; ++k ) {
				g0[k] += C_i * Q_i[k];
			}
		}
	}
}

void CSMOptimizer::swapIndex( int i, int j )
{
	Q->SwapIndex( i, j );
	swap( g[i], g[j] );
	swap( g0[i], g0[j] );
	swap( alpha[i], alpha[j] );
	swap( C[i], C[j] );
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

	if( !unshrink && gMax1 + gMax2 <= tolerance * 10 ) {
		unshrink = true;
		reconstructGradient();
		activeSize = l;
		if( log != nullptr ) {
			*log << "*";
		}
	}

	for( int i = 0; i < activeSize; ++i ) {
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
			auto Q_i = Q->GetColumn( i, activeSize );
			for( int j = 0; j < activeSize; ++j ) {
				if( alphaStatus[j] == AS_Free ) {
					g[i] += alpha[j] * Q_i[j];
				}
			}
		}
	} else {
		for( int i = 0; i < activeSize; ++i ) {
			if( alphaStatus[i] == AS_Free ) {
				auto Q_i = Q->GetColumn( i, l );
				double alpha_i = alpha[i];
				for( int j = activeSize; j < l; ++j ) {
					g[j] += alpha_i * Q_i[j];
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

