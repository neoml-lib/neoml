/* Copyright © 2017-2021 ABBYY Production LLC

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

#include <common.h>
#pragma hdrstop

#include <OneVersusOneModel.h>

namespace NeoML {

REGISTER_NEOML_MODEL( COneVersusOneModel, OneVersusOneModelName )

// returns class count for given classifier count size
// classifier count must be equal to classCount * (classCount - 1) / 2
static inline int getClassCount( int classifierCount )
{
	int result = static_cast<int>( ceil( sqrt( static_cast<double>( 2 * classifierCount ) ) ) );
	NeoPresume( result * ( result - 1 ) == classifierCount * 2 );
	return result;
}

//---------------------------------------------------------------------------------------------------------

// Optimization method which finds a vector of probabilities
// from classCount * (classCount - 1) / 2 base classifier predictions

// article: https://www.csie.ntu.edu.tw/~cjlin/papers/svmprob/svmprob.pdf (section 4. Our Second Approach)

// In short: we're given classCount * (classCount - 1) / 2  for each pair of classes (pred)
// We've got to find vector of probabilities (prob) of classCount length which meets several conditions:
//
//     1. It's a probability distribtuion (consisting of non-negative numbers and their sum is equal to 1)
//     2. It approximates given predictions (pred[i][j] == prob[i] / (prob[i] + prob[j])

// calculates products which will be used in optimization
static inline void calcProducts( const CArray<CArray<float>>& mat, const CArray<float>& prob,
	CArray<float>& matByProb, float& probByMatByProb )
{
	const int classCount = mat.Size();
	NeoPresume( classCount > 1 );
	NeoPresume( matByProb.Size() == classCount );

	probByMatByProb = 0.f;
	for( int i = 0; i < classCount; ++i ) {
		matByProb[i] = 0.f;
		for( int j = 0; j < classCount; ++j ) {
			matByProb[i] += mat[i][j] * prob[j];
		}
		probByMatByProb += prob[i] * matByProb[i];
	}
}

// checks whether the algorithm has converged
static inline bool hasConverged( const CArray<float>& matByProb, float probByMatByProb, float eps )
{
	const int classCount = matByProb.Size();
	NeoPresume( classCount > 1 );
	for( int i = 0; i < classCount; ++i ) {
		const float diff = fabs( probByMatByProb - matByProb[i] );
		if( diff > eps ) {
			return false;
		}
	}
	return true;
}

// fills prob with approximated probabilities based on pred
// pred is a matrix of size classCount x classCount (r in the article)
static void findProb( const CArray<CArray<float>>& pred, CArray<float>& prob )
{
	const int classCount = pred.Size();
	NeoAssert( classCount > 1 );

	// Fill prob with initial distribution (p in the article)
	prob.DeleteAll();
	prob.Add( 1.f / classCount, classCount );

	// matrix used in optimization task (Q in the article)
	CArray<CArray<float>> mat;
	mat.SetSize( classCount );
	for( int row = 0; row < classCount; ++row ) {
		mat[row].SetSize( classCount );
		mat[row][row] = 0.f;
	}
	for( int row = 0; row < classCount; ++row ) {
		for( int col = 0; col < classCount; ++col ) {
			if( row == col ) {
				continue;
			}
			mat[row][row] += pred[col][row] * pred[col][row];
			mat[row][col] = -pred[row][col] * pred[col][row];
		}
	}

	CArray<float> matByProb; // Q*p in terms of article
	matByProb.SetSize( classCount );
	const float eps = 1e-2f / classCount;
	const int maxIter = 50;

	for( int iter = 0; iter < maxIter; ++iter ) {
		float probByMatByProb;  // p*Q*p in terms of article
		calcProducts( mat, prob, matByProb, probByMatByProb );

		if( hasConverged( matByProb, probByMatByProb, eps ) ) {
			break;
		}

		// Perform step (update once each of prob values)
		for( int i = 0; i < classCount; ++i ) {
			const float diff = ( probByMatByProb - matByProb[i] ) / mat[i][i];
			prob[i] += diff;
			const float newSum = 1.f + diff;
			probByMatByProb = ( probByMatByProb + diff * ( diff * mat[i][i] + 2 * matByProb[i] ) ) / ( newSum * newSum );
			for( int j = 0; j < classCount; ++j ) {
				matByProb[j] = ( matByProb[j] + diff * mat[i][j] ) / newSum;
				prob[j] /= newSum; // normalize prob vector back to the sum of 1
			}
		}
	}
}

//---------------------------------------------------------------------------------------------------------

COneVersusOneModel::COneVersusOneModel( CObjectArray<IModel>& _classifiers ) :
	classCount( getClassCount( _classifiers.Size() ) )
{
	NeoAssert( !_classifiers.IsEmpty() );
	_classifiers.MoveTo( classifiers );
}

bool COneVersusOneModel::Classify( const CFloatVectorDesc& data, CClassificationResult& result ) const
{
	CArray<CArray<float>> pred;
	pred.SetSize( classCount );
	for( int i = 0; i < classCount; ++i ) {
		pred[i].Add( 0.f, classCount );
	}

	int classifierIndex = 0;
	for( int i = 0; i < classCount - 1; ++i ) {
		for( int j = i + 1; j < classCount; ++j ) {
			CClassificationResult subresult;
			NeoAssert( classifiers[classifierIndex++]->Classify( data, subresult ) );
			NeoPresume( subresult.Probabilities.Size() == 2 );
			pred[i][j] = static_cast<float>( subresult.Probabilities[0].GetValue() );
			pred[j][i] = static_cast<float>( subresult.Probabilities[1].GetValue() );
		}
	}

	CArray<float> prob;
	findProb( pred, prob );
	result.ExceptionProbability = CClassificationProbability( 0 );
	result.Probabilities.SetBufferSize( classCount );
	result.PreferredClass = 0;
	for( int i = 0; i < classCount; ++i ) {
		result.Probabilities.Add( CClassificationProbability( static_cast<double>( prob[i] ) ) );
		if( prob[i] > prob[result.PreferredClass] ) {
			result.PreferredClass = i;
		}
	}

	return true;
}

static const int oneVersusOneVersion = 0;

void COneVersusOneModel::Serialize( CArchive& archive )
{
	archive.SerializeVersion( oneVersusOneVersion );
	if( archive.IsStoring() ) {
		archive << classifiers.Size();
		for( int i = 0; i < classifiers.Size(); ++i ) {
			archive << CString( GetModelName( classifiers[i] ) );
			classifiers[i]->Serialize( archive );
		}
	} else if( archive.IsLoading() ) {
		int size = 0;
		archive >> size;
		classifiers.SetSize( size );
		for( int i = 0; i < classifiers.Size(); ++i ) {
			CString name;
			archive >> name;
			classifiers[i] = CreateModel<IModel>( name );
			classifiers[i]->Serialize( archive );
		}
		classCount = getClassCount( classifiers.Size() );
	}
}

} // namespace NeoML
