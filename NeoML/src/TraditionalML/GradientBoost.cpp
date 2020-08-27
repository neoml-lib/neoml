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

#include <common.h>
#pragma hdrstop

#include <NeoML/TraditionalML/GradientBoost.h>
#include <GradientBoostModel.h>
#include <RegressionTreeModel.h>
#include <GradientBoostFullProblem.h>
#include <GradientBoostFastHistProblem.h>
#include <GradientBoostFullTreeBuilder.h>
#include <GradientBoostFastHistTreeBuilder.h>
#include <ProblemWrappers.h>
#include <NeoMathEngine/OpenMP.h>

namespace NeoML {

const double MaxExpArgument = 30; // the maximum argument for an exponent

IGradientBoostModel::~IGradientBoostModel()
{
}

IGradientBoostRegressionModel::~IGradientBoostRegressionModel()
{
}

IRegressionTreeModel::~IRegressionTreeModel()
{
}

// Loss function interface
class IGradientBoostingLossFunction : public virtual IObject {
public:
	// Calculates function gradient
	virtual void CalcGradientAndHessian( const CArray< CArray<double> >& predicts, const CArray< CArray<double> >& answers,
		CArray< CArray<double> >& gradient, CArray< CArray<double> >& hessian ) const = 0;
};

//------------------------------------------------------------------------------------------------------------

// Binomial loss function
class CGradientBoostingBinomialLossFunction : public IGradientBoostingLossFunction {
public:
	// IGradientBoostingLossFunction interface methods
	virtual void CalcGradientAndHessian( const CArray< CArray<double> >& predicts, const CArray< CArray<double> >& answers,
		CArray< CArray<double> >& gradient, CArray< CArray<double> >& hessian ) const;
};

void CGradientBoostingBinomialLossFunction::CalcGradientAndHessian( const CArray< CArray<double> >& predicts,
	const CArray< CArray<double> >& answers, CArray< CArray<double> >& gradients, CArray< CArray<double> >& hessians ) const
{
	NeoAssert( predicts.Size() == answers.Size() );

	gradients.SetSize( predicts.Size() );
	hessians.SetSize( predicts.Size() );

	for( int i = 0; i < predicts.Size(); i++ ) {
		gradients[i].Empty();
		hessians[i].Empty();
		for( int j = 0; j < predicts[i].Size(); j++ ) {
			const double pred = 1.0f / ( 1.0f + exp( min( -predicts[i][j], MaxExpArgument ) ) );
			gradients[i].Add( static_cast<double>( pred - answers[i][j] ) );
			hessians[i].Add( static_cast<double>( max( pred * ( 1.0 - pred ), 1e-16 ) ) );
		}
	}
}

//------------------------------------------------------------------------------------------------------------

// Exponential loss function (similar to AdaBoost)
class CGradientBoostingExponentialLossFunction : public IGradientBoostingLossFunction {
public:
	// IGradientBoostingLossFunction interface methods
	virtual void CalcGradientAndHessian( const CArray< CArray<double> >& predicts, const CArray< CArray<double> >& answers,
		CArray< CArray<double> >& gradient, CArray< CArray<double> >& hessian ) const;
};

void CGradientBoostingExponentialLossFunction::CalcGradientAndHessian( const CArray< CArray<double> >& predicts,
	const CArray< CArray<double> >& answers, CArray< CArray<double> >& gradients, CArray< CArray<double> >& hessians ) const
{
	NeoAssert( predicts.Size() == answers.Size() );

	gradients.SetSize( predicts.Size() );
	hessians.SetSize( predicts.Size() );

	for( int i = 0; i < predicts.Size(); i++ ) {
		gradients[i].Empty();
		hessians[i].Empty();
		for( int j = 0; j < predicts[i].Size(); j++ ) {
			const double temp = -( 2 * answers[i][j] - 1 );
			const double tempExp = exp( min( temp * predicts[i][j], MaxExpArgument ) );
			gradients[i].Add( static_cast<double>( temp * tempExp ) );
			hessians[i].Add( static_cast<double>( temp * temp * tempExp ) );
		}
	}
}

//------------------------------------------------------------------------------------------------------------

// Smoothed square hinge function
class CGradientBoostingSquaredHinge : public IGradientBoostingLossFunction {
public:
	// IGradientBoostingLossFunction interface methods
	virtual void CalcGradientAndHessian( const CArray< CArray<double> >& predicts, const CArray< CArray<double> >& answers,
		CArray< CArray<double> >& gradient, CArray< CArray<double> >& hessian ) const;
};

void CGradientBoostingSquaredHinge::CalcGradientAndHessian( const CArray< CArray<double> >& predicts,
	const CArray< CArray<double> >& answers, CArray< CArray<double> >& gradients, CArray< CArray<double> >& hessians ) const
{
	NeoAssert( predicts.Size() == answers.Size() );

	gradients.SetSize( predicts.Size() );
	hessians.SetSize( predicts.Size() );

	for( int i = 0; i < predicts.Size(); i++ ) {
		gradients[i].Empty();
		hessians[i].Empty();
		for( int j = 0; j < predicts[i].Size(); j++ ) {
			const double t = -( 2 * answers[i][j] - 1 );

			if( t * predicts[i][j] < 1 ) {
				gradients[i].Add( static_cast<double>( 2 * t * ( t * predicts[i][j] - 1 ) ) );
				hessians[i].Add( static_cast<double>( 2 * t * t ) );
			} else {
				gradients[i].Add( 0.0 );
				hessians[i].Add( 1e-16 );
			}
		}
	}
}

//------------------------------------------------------------------------------------------------------------

// Quadratic loss function for classification and regression
class CGradientBoostingSquareLoss : public IGradientBoostingLossFunction {
public:
	// IGradientBoostingLossFunction interface methods
	virtual void CalcGradientAndHessian( const CArray< CArray<double> >& predicts, const CArray< CArray<double> >& answers,
		CArray< CArray<double> >& gradient, CArray< CArray<double> >& hessian ) const;
};

void CGradientBoostingSquareLoss::CalcGradientAndHessian( const CArray< CArray<double> >& predicts,
	const CArray< CArray<double> >& answers, CArray< CArray<double> >& gradients, CArray< CArray<double> >& hessians ) const
{
	NeoAssert( predicts.Size() == answers.Size() );

	gradients.SetSize( predicts.Size() );
	hessians.SetSize( predicts.Size() );

	for( int i = 0; i < predicts.Size(); i++ ) {
		gradients[i].Empty();
		hessians[i].Empty();
		for( int j = 0; j < predicts[i].Size(); j++ ) {
			gradients[i].Add( static_cast<double>( predicts[i][j] - answers[i][j] ) );
			hessians[i].Add( static_cast<double>( 1.0 ) );
		}
	}
}

//------------------------------------------------------------------------------------------------------------

// Generates an array of random K numbers in the [0, N) range
static void generateRandomArray( CRandom& random, int n, int k, CArray<int>& result )
{
	NeoAssert( k <= n );
	NeoAssert( 1 <= k );

	result.Empty();
	result.SetBufferSize( n );
	for( int i = 0; i < n; i++ ) {
		result.Add( i );
	}

	if( k == n ) {
		return;
	}

	for( int i = 0; i < k; i++ ) {
		// Choose a random number from [i, n - 1] range
		const int index = random.UniformInt( i, n - 1 );
		swap( result[i], result[index] );
	}
	result.SetSize( k );
	result.QuickSort< Ascending<int> >();
}

//------------------------------------------------------------------------------------------------------------

#if FINE_PLATFORM( FINE_IOS )
	// No OpenMP available for iOS, so working in one thread
	static inline CGradientBoost::CParams processParams( const CGradientBoost::CParams& params )
	{
		CGradientBoost::CParams result = params;
		result.ThreadCount = 1;
		return result;
	}
#elif FINE_PLATFORM( FINE_WINDOWS ) || FINE_PLATFORM( FINE_LINUX ) || FINE_PLATFORM( FINE_ANDROID ) || FINE_PLATFORM( FINE_DARWIN )
	static inline CGradientBoost::CParams processParams( const CGradientBoost::CParams& params ) { return params; }
#else
	#error Unknown platform
#endif

CGradientBoost::CGradientBoost( const CParams& _params ) :
	params( processParams( _params ) ),
	logStream( 0 )
{
	NeoAssert( params.IterationsCount > 0 );
	NeoAssert( 0 <= params.Subsample && params.Subsample <= 1 );
	NeoAssert( 0 <= params.Subfeature && params.Subfeature <= 1 );
	NeoAssert( params.MaxTreeDepth >= 0 );
	NeoAssert( params.MaxNodesCount >= 0 || params.MaxNodesCount == NotFound );
	NeoAssert( params.PruneCriterionValue >= 0 );
	NeoAssert( params.ThreadCount > 0 );
	NeoAssert( params.MinSubsetWeight >= 0 );
}

CGradientBoost::~CGradientBoost()
{
}

CPtr<IGradientBoostRegressionModel> CGradientBoost::TrainRegression(
	const IBaseRegressionProblem& problem )
{
	if( logStream != nullptr ) {
		*logStream << "\nGradient boost regression training started:\n";
	}

	CPtr<const IMultivariateRegressionProblem> multivariate =
		dynamic_cast<const IMultivariateRegressionProblem*>( &problem );
	if( multivariate == nullptr ) {
		multivariate = FINE_DEBUG_NEW CMultivariateRegressionOverUnivariate(
			dynamic_cast<const IRegressionProblem*>( &problem ) );
	}

	return train( multivariate, createLossFunction() ).Ptr();
}

CPtr<IRegressionModel> CGradientBoost::TrainRegression( const IRegressionProblem& problem )
{
	if( logStream != nullptr ) {
		*logStream << "\nGradient boost regression training started:\n";
	}

	CPtr<const IMultivariateRegressionProblem> multivariate =
		FINE_DEBUG_NEW CMultivariateRegressionOverUnivariate( &problem );

	return train( multivariate, createLossFunction() ).Ptr();
}

CPtr<IModel> CGradientBoost::Train( const IProblem& problem )
{
	if( logStream != nullptr ) {
		*logStream << "\nGradient boost training started:\n";
	}

	CPtr<const IMultivariateRegressionProblem> multivariate;
	if( problem.GetClassCount() == 2 ) {
		multivariate = FINE_DEBUG_NEW CMultivariateRegressionOverBinaryClassification( &problem );
	} else {
		multivariate = FINE_DEBUG_NEW CMultivariateRegressionOverClassification( &problem );
	}

	return train( multivariate, createLossFunction() ).Ptr();
}

// Trains a model
CPtr<CGradientBoostModel> CGradientBoost::train(
	const IMultivariateRegressionProblem* _problem,
	IGradientBoostingLossFunction* lossFunction )
{
	NeoAssert( _problem != nullptr && lossFunction != nullptr );

	// create view without null weights over original problem
	CPtr<const IMultivariateRegressionProblem> problem = 
		FINE_DEBUG_NEW CMultivariateRegressionProblemNotNullWeightsView( _problem );
	CArray<CGradientBoostEnsemble> models; // the final models ensemble (ensembles are used for multi-class classification)
	initialize( problem->GetValueSize(), problem->GetVectorCount(),
		problem->GetFeatureCount(), models );

	try {
		// Create a tree builder
		createTreeBuilder( problem );

		// Every new tree is trained on a new problem
		for( int i = 0; i < params.IterationsCount; i++ ) {
			if( logStream != nullptr ) {
				*logStream << "\nBoost iteration " << i << ":\n";
			}

			// One gradient boosting step
			CObjectArray<IRegressionModel> curIterationModels; // a new model for multi-class classification
			executeStep( *lossFunction, problem, models, curIterationModels );

			for( int j = 0; j < curIterationModels.Size(); j++ ) {
				models[j].Add( curIterationModels[j] );
			}
		}
	} catch( ... ) {
		destroyTreeBuilder(); // return to the initial state
		throw;
	}
	destroyTreeBuilder();

	return FINE_DEBUG_NEW CGradientBoostModel( models, params.LearningRate, params.LossFunction );
}

// Creates a tree builder depending on the problem type
void CGradientBoost::createTreeBuilder( const IMultivariateRegressionProblem* problem )
{
	switch( params.TreeBuilder ) {
		case GBTB_Full:
		{
			CGradientBoostFullTreeBuilder::CParams builderParams;
			builderParams.L1RegFactor = params.L1RegFactor;
			builderParams.L2RegFactor = params.L2RegFactor;
			builderParams.MinSubsetHessian = 1e-3f;
			builderParams.ThreadCount = params.ThreadCount;
			builderParams.MaxTreeDepth = params.MaxTreeDepth;
			builderParams.MaxNodesCount = params.MaxNodesCount;
			builderParams.PruneCriterionValue = params.PruneCriterionValue;
			builderParams.MinSubsetWeight = params.MinSubsetWeight;
			fullTreeBuilder = FINE_DEBUG_NEW CGradientBoostFullTreeBuilder( builderParams, logStream );
			fullProblem = FINE_DEBUG_NEW CGradientBoostFullProblem( params.ThreadCount, problem,
				usedVectors, usedFeatures, featureNumbers );
			break;
		}
		case GBTB_FastHist:
		{
			CGradientBoostFastHistTreeBuilder::CParams builderParams;
			builderParams.L1RegFactor = params.L1RegFactor;
			builderParams.L2RegFactor = params.L2RegFactor;
			builderParams.MinSubsetHessian = 1e-3f;
			builderParams.ThreadCount = params.ThreadCount;
			builderParams.MaxTreeDepth = params.MaxTreeDepth;
			builderParams.MaxNodesCount = params.MaxNodesCount;
			builderParams.PruneCriterionValue = params.PruneCriterionValue;
			builderParams.MaxBins = params.MaxBins;
			builderParams.MinSubsetWeight = params.MinSubsetWeight;
			fastHistTreeBuilder = FINE_DEBUG_NEW CGradientBoostFastHistTreeBuilder( builderParams, logStream );
			fastHistProblem = FINE_DEBUG_NEW CGradientBoostFastHistProblem( params.ThreadCount, params.MaxBins,
				*problem, usedVectors, usedFeatures );
			break;
		}
		default:
			NeoAssert( false );
	}
}

// Destroys a tree builder
void CGradientBoost::destroyTreeBuilder()
{
	fullTreeBuilder.Release();
	fullProblem.Release();
	fastHistTreeBuilder.Release();
	fastHistProblem.Release();
}

// Creates a loss function based on CParam.LossFunction
CPtr<IGradientBoostingLossFunction> CGradientBoost::createLossFunction() const
{
	switch( params.LossFunction ) {
		case LF_Binomial:
			return FINE_DEBUG_NEW CGradientBoostingBinomialLossFunction();
			break;
		case LF_Exponential:
			return FINE_DEBUG_NEW CGradientBoostingExponentialLossFunction();
			break;
		case LF_SquaredHinge:
			return FINE_DEBUG_NEW CGradientBoostingSquaredHinge();
			break;
		case LF_L2:
			return FINE_DEBUG_NEW CGradientBoostingSquareLoss();
			break;
		default:
			NeoAssert( false );
			return 0;
	}
}

// Initializes the algorithm
void CGradientBoost::initialize( int modelCount, int vectorCount, int featureCount, CArray<CGradientBoostEnsemble>& models )
{
	NeoAssert( modelCount >= 1 );
	NeoAssert( vectorCount > 0 );
	NeoAssert( featureCount > 0 );

	models.SetSize( modelCount );

	predictCache.DeleteAll();
	predictCache.SetSize( modelCount );
	CPredictionCacheItem item;
	item.Step = 0;
	item.Value = 0;
	for( int i = 0; i < predictCache.Size(); i++ ) {
		predictCache[i].Add( item, vectorCount );
	}

	predicts.SetSize( modelCount );
	answers.SetSize( modelCount );
	gradients.SetSize( modelCount );
	hessians.SetSize( modelCount );
	if( params.Subsample == 1.0 ) {
		usedVectors.DeleteAll();
		for( int i = 0; i < vectorCount; i++ ) {
			usedVectors.Add( i );
		}
	}
	if( params.Subfeature == 1.0 ) {
		usedFeatures.DeleteAll();
		featureNumbers.DeleteAll();
		for(int i = 0; i < featureCount; i++ ) {
			usedFeatures.Add( i );
			featureNumbers.Add( i );
		}
	}
}

// Performs gradient boosting iteration
// On a sub-problem of the first problem using cache
void CGradientBoost::executeStep( IGradientBoostingLossFunction& lossFunction,
	const IMultivariateRegressionProblem* problem,
	const CArray<CGradientBoostEnsemble>& models, CObjectArray<IRegressionModel>& curModels )
{
	NeoAssert( !models.IsEmpty() );
	NeoAssert( curModels.IsEmpty() );
	NeoAssert( problem != nullptr );

	const int vectorCount = problem->GetVectorCount();
	const int featureCount = problem->GetFeatureCount();

	if( params.Subsample < 1.0 ) {
		generateRandomArray( params.Random != nullptr ? *params.Random : defaultRandom, vectorCount,
			max( static_cast<int>( vectorCount * params.Subsample ), 1 ), usedVectors );
	}
	if( params.Subfeature < 1.0 ) {
		generateRandomArray( params.Random != nullptr ? *params.Random : defaultRandom, featureCount,
			max( static_cast<int>( featureCount * params.Subfeature ), 1 ), usedFeatures );
		
		if( featureNumbers.Size() != featureCount ) {
			featureNumbers.SetSize( featureCount );
		}
		for( int i = 0; i < featureCount; ++i ) {
			featureNumbers[i] = NotFound;
		}
		for( int i = 0; i < usedFeatures.Size(); ++i ) {
			featureNumbers[usedFeatures[i]] = i;
		}
	}

	const int curStep = models[0].Size();

	for( int i = 0; i < predicts.Size(); i++ ) {
		predicts[i].SetSize( usedVectors.Size() );
		answers[i].SetSize( usedVectors.Size() );
		gradients[i].Empty();
		hessians[i].Empty();
	}

	// Build the current model predictions
	buildPredictions( *problem, models, curStep );

	// The vectors in the regression value are partial derivatives of the loss function
	// The tree built for this problem will decrease the loss function value
	lossFunction.CalcGradientAndHessian( predicts, answers, gradients, hessians );

	// Add the vector weights and calculate the total
	CArray<double> gradientsSum;
	gradientsSum.Add( 0, gradients.Size() );
	CArray<double> hessiansSum;
	hessiansSum.Add( 0, gradients.Size() );
	CArray<float> weights;
	weights.SetSize( usedVectors.Size() );

	float weightsSum = 0;
	for( int i = 0; i < usedVectors.Size(); i++ ) {
		weights[i] = static_cast<float>( problem->GetVectorWeight( usedVectors[i] ) );
		weightsSum += weights[i];
	}

	for( int i = 0; i < gradients.Size(); i++ ) {
		for( int j = 0; j < usedVectors.Size(); j++ ) {
			gradients[i][j] = gradients[i][j] * weights[j];
			gradientsSum[i] += gradients[i][j];
			hessians[i][j] = hessians[i][j] * weights[j];
			hessiansSum[i] += hessians[i][j];
		}
	}

	if( curStep == 0 || params.Subfeature != 1.0 || params.Subsample != 1.0 ) {
		// The sub-problem data has changed, reload it
		if( fullProblem != nullptr ) {
			fullProblem->Update();
		}
	}

	for( int i = 0; i < gradients.Size(); i++ ) {
		if( logStream != nullptr ) {
			*logStream << "GradientSum = " << gradientsSum[i]
				<< " HessianSum = " << hessiansSum[i]
				<< "\n";
		}
		CPtr<IRegressionModel> model;
		if( fullTreeBuilder != nullptr ) {
			model = fullTreeBuilder->Build( *fullProblem, gradients[i], gradientsSum[i], hessians[i], hessiansSum[i], weights, weightsSum );
		} else {
			model = fastHistTreeBuilder->Build( *fastHistProblem, gradients[i], hessians[i], weights );
		}
		curModels.Add( model );
	}
}

// Builds the ensemble predictions for a set of vectors
void CGradientBoost::buildPredictions( const IMultivariateRegressionProblem& problem, const CArray<CGradientBoostEnsemble>& models, int curStep )
{
	CSparseFloatMatrixDesc matrix = problem.GetMatrix();
	NeoAssert( matrix.Height == problem.GetVectorCount() );
	NeoAssert( matrix.Width == problem.GetFeatureCount() );

	NEOML_OMP_NUM_THREADS( params.ThreadCount )
	{
		int index = 0;
		int count = 0;
		if( OmpGetTaskIndexAndCount( usedVectors.Size(), index, count ) ) {
			for( int i = 0; i < count; i++ ) {
				const int usedVector = usedVectors[index];
				const CFloatVector value = problem.GetValue( usedVectors[index] );
				CSparseFloatVectorDesc vector;
				matrix.GetRow( usedVector, vector );

				for( int j = 0; j < models.Size(); j++ ) {
					predictCache[j][usedVector].Value += CGradientBoostModel::PredictRaw( models[j], predictCache[j][usedVector].Step,
						params.LearningRate, vector );
					predictCache[j][usedVector].Step = curStep;
					predicts[j][index] = predictCache[j][usedVector].Value;
					answers[j][index] = value[j];
				}
				index++;
			}
		}
	}
}

} // namespace NeoML
