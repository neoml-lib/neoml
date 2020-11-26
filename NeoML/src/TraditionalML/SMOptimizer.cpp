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
	const float* GetBinaryClasses() const { return classes.GetPtr(); }

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
	gradient.Add(-1., l ); // the target function gradient
	g = gradient.GetPtr();
	_alpha.Empty();
	_alpha.Add( 0., l ); // the support vectors coefficients
	alpha = _alpha.GetPtr();

	int t;
	int counter = min( l, 1000 );
	for(t = 0; t < maxIter; t++) {
		if( --counter == 0 ) {
			counter = min( l, 1000 );
			// log progress
			if( log != nullptr ) {
				*log << ".";
			}
		}

		// Find a pair of coefficients that violate Kuhn - Tucker conditions most of all
		int i, j; 
		if( selectWorkingSet( i, j ) ) {
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
}

// reconstruct inactive elements of G from G_bar and free variables
void CSMOptimizer::reconstructGradient()
{
	if( activeSize == l ) {
		return;
	}
	NeoAssert( false );

	int freeCount = 0;
	for( int j = activeSize; j < l; ++j ) {
		g[j] = g0[j] - 1;
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
					g[i] += alpha[j] * Q_i[j];
			}
		}
	} else {
		for( int i = 0; i < activeSize; ++i ) {
			if( alphaStatus[i] == FREE ) {
				auto Q_i = Q->GetColumn( i );
				for( int j = activeSize; j<l; ++j )
					g[j] += alpha[i] * Q_i[j];
			}
		}
	}
}

// return i,j such that
// i: maximizes -y_i * grad(f)_i, i in I_up(\alpha)
// j: minimizes the decrease of obj value
//    (if quadratic coefficient <= 0, replace it with tau)
//    -y_j*grad(f)_j < -y_i*grad(f)_i, j in I_low(\alpha)
bool CSMOptimizer::selectWorkingSet( int& outI, int& outJ ) const
{
	double Gmax = -Inf;
	double Gmax2 = -Inf;
	int Gmax_idx = -1;
	int Gmin_idx = -1;
	double obj_diff_min = Inf;

	for(int t=0;t<l;t++)
		if( y[t] ==+1 )
		{
			if(alpha[t] < C[t])
				if(-g[t] >= Gmax)
				{
					Gmax = -g[t];
					Gmax_idx = t;
				}
		}
		else
		{
			if(alpha[t] > 0)
				if(g[t] >= Gmax)
				{
					Gmax = g[t];
					Gmax_idx = t;
				}
		}

	int i = Gmax_idx;
	const float *Q_i = NULL;
	if(i != -1) 
		Q_i = Q->GetColumn(i);

	for(int j=0;j<l;j++)
	{
		if( y[j] ==+1 )
		{
			if (alpha[j] > 0)
			{
				double grad_diff=Gmax+g[j];
				if (g[j] >= Gmax2)
					Gmax2 = g[j];
				if (grad_diff > 0)
				{
					double obj_diff;
					double quad_coef = QD[i]+QD[j]-2.0*y[i]*Q_i[j];
					if (quad_coef > 0)
						obj_diff = -(grad_diff*grad_diff)/quad_coef;
					else
						obj_diff = -(grad_diff*grad_diff)/Tau;

					if (obj_diff <= obj_diff_min)
					{
						Gmin_idx=j;
						obj_diff_min = obj_diff;
					}
				}
			}
		}
		else
		{
			if (alpha[j] < C[j])
			{
				double grad_diff= Gmax-g[j];
				if (-g[j] >= Gmax2)
					Gmax2 = -g[j];
				if (grad_diff > 0)
				{
					double obj_diff;
					double quad_coef = QD[i]+QD[j]+2.0*y[i]*Q_i[j];
					if (quad_coef > 0)
						obj_diff = -(grad_diff*grad_diff)/quad_coef;
					else
						obj_diff = -(grad_diff*grad_diff)/Tau;

					if (obj_diff <= obj_diff_min)
					{
						Gmin_idx=j;
						obj_diff_min = obj_diff;
					}
				}
			}
		}
	}

	if(Gmax+Gmax2 < tolerance || Gmin_idx == -1)
		return 1;

	outI = Gmax_idx;
	outJ = Gmin_idx;
	return 0;
}

// Optimizes the target function by changing the alpha_i and alpha_j coefficient
// The optimal values are calculated analytically
void CSMOptimizer::optimizePair( int i, int j ) const
{	
	const float* Q_i = Q->GetColumn(i);
	const float* Q_j = Q->GetColumn(j);

	double C_i = C[i];
	double C_j = C[j];
	double oldAlpha_i = alpha[i];
	double oldAlpha_j = alpha[j];

	if( y[i] != y[j] ) {
		double quadCoef = QD[i] + QD[j] + 2 * Q_i[j];
		if (quadCoef <= 0) {
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
		if (quadCoef <= 0)
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
	for(int k = 0; k < l; k++) {
		g[k] += Q_i[k] * deltaAlpha_i + Q_j[k] * deltaAlpha_j;
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

