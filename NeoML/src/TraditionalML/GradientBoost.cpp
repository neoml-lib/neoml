/* Copyright © 2017-2023 ABBYY

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
#include <NeoML/TraditionalML/GradientBoostQuickScorer.h>
#include <GradientBoostModel.h>
#include <RegressionTree.h>
#include <ProblemWrappers.h>
#include <GradientBoostFullProblem.h>
#include <GradientBoostFastHistProblem.h>
#include <GradientBoostFullTreeBuilder.h>
#include <GradientBoostFastHistTreeBuilder.h>
#include <GradientBoostThreadTask.h>
#include <NeoMathEngine/ThreadPool.h>

namespace NeoML {

const double MaxExpArgument = 30; // the maximum argument for an exponent

IGradientBoostModel::~IGradientBoostModel() = default;

IGradientBoostRegressionModel::~IGradientBoostRegressionModel() = default;

IRegressionTreeNode::~IRegressionTreeNode() = default;

// Loss function interface
class IGradientBoostingLossFunction : public virtual IObject {
public:
	// Calculates function gradient
	virtual void CalcGradientAndHessian( const CArray<CArray<double>>& predicts, const CArray<CArray<double>>& answers,
		CArray<CArray<double>>& gradient, CArray<CArray<double>>& hessian ) const = 0;

	// Calculates loss
	virtual double CalcLossMean( const CArray<CArray<double>>& predicts, const CArray<CArray<double>>& answers ) const = 0;
};

//------------------------------------------------------------------------------------------------------------

// Binomial loss function
class CGradientBoostingBinomialLossFunction : public IGradientBoostingLossFunction {
public:
	// IGradientBoostingLossFunction interface methods
	void CalcGradientAndHessian( const CArray< CArray<double> >& predicts, const CArray< CArray<double> >& answers,
		CArray< CArray<double> >& gradient, CArray< CArray<double> >& hessian ) const override;

	double CalcLossMean( const CArray< CArray<double> >& predicts, const CArray< CArray<double> >& answers ) const override;
};

void CGradientBoostingBinomialLossFunction::CalcGradientAndHessian( const CArray< CArray<double> >& predicts,
	const CArray< CArray<double> >& answers, CArray< CArray<double> >& gradients, CArray< CArray<double> >& hessians ) const
{
	NeoAssert( predicts.Size() == answers.Size() );

	gradients.SetSize( predicts.Size() );
	hessians.SetSize( predicts.Size() );

	for( int i = 0; i < predicts.Size(); i++ ) {
		gradients[i].SetSize( predicts[i].Size() );
		hessians[i].SetSize( predicts[i].Size() );
		for( int j = 0; j < predicts[i].Size(); j++ ) {
			const double pred = 1.0f / ( 1.0f + exp( min( -predicts[i][j], MaxExpArgument ) ) );
			gradients[i][j] = static_cast<double>( pred - answers[i][j] );
			hessians[i][j] = static_cast<double>( max( pred * ( 1.0 - pred ), 1e-16 ) );
		}
	}
}

double CGradientBoostingBinomialLossFunction::CalcLossMean( const CArray< CArray<double> >& predicts,
	const CArray< CArray<double> >& answers ) const
{
	NeoAssert( predicts.Size() == answers.Size() );

	double overallSum = 0;
	auto getMean = []( double sum, int n ) { return n != 0 ? sum / static_cast<double>( n ) : 0; };
	for( int i = 0; i < predicts.Size(); ++i ) {
		double sum = 0;
		for( int j = 0; j < predicts[i].Size(); ++j ) {
			sum += log1p( exp( min( -predicts[i][j], MaxExpArgument ) ) ) - predicts[i][j] * answers[i][j];
		}
		overallSum += getMean( sum, predicts[i].Size() );
	}

	return getMean( overallSum, predicts.Size() );
}

//------------------------------------------------------------------------------------------------------------

// Exponential loss function (similar to AdaBoost)
class CGradientBoostingExponentialLossFunction : public IGradientBoostingLossFunction {
public:
	// IGradientBoostingLossFunction interface methods
	void CalcGradientAndHessian( const CArray< CArray<double> >& predicts, const CArray< CArray<double> >& answers,
		CArray< CArray<double> >& gradient, CArray< CArray<double> >& hessian ) const override;

	double CalcLossMean( const CArray< CArray<double> >& predicts, const CArray< CArray<double> >& answers ) const override;
};

void CGradientBoostingExponentialLossFunction::CalcGradientAndHessian( const CArray< CArray<double> >& predicts,
	const CArray< CArray<double> >& answers, CArray< CArray<double> >& gradients, CArray< CArray<double> >& hessians ) const
{
	NeoAssert( predicts.Size() == answers.Size() );

	gradients.SetSize( predicts.Size() );
	hessians.SetSize( predicts.Size() );

	for( int i = 0; i < predicts.Size(); i++ ) {
		gradients[i].SetSize( predicts[i].Size() );
		hessians[i].SetSize( predicts[i].Size() );
		for( int j = 0; j < predicts[i].Size(); j++ ) {
			const double temp = -( 2 * answers[i][j] - 1 );
			const double tempExp = exp( min( temp * predicts[i][j], MaxExpArgument ) );
			gradients[i][j] = static_cast<double>( temp * tempExp );
			hessians[i][j] = static_cast<double>( temp * temp * tempExp );
		}
	}
}

double CGradientBoostingExponentialLossFunction::CalcLossMean( const CArray< CArray<double> >& predicts,
	const CArray< CArray<double> >& answers ) const
{
	NeoAssert( predicts.Size() == answers.Size() );

	double overallSum = 0;
	auto getMean = []( double sum, int n ) { return n != 0 ? sum / static_cast<double>( n ) : 0; };
	for( int i = 0; i < predicts.Size(); ++i ) {
		double sum = 0;
		for( int j = 0; j < predicts[i].Size(); ++j ) {
			sum += exp( min( ( 1.0 - 2.0 * answers[i][j] ) * predicts[i][j], MaxExpArgument ) );
		}
		overallSum += getMean( sum, predicts[i].Size() );
	}

	return getMean( overallSum, predicts.Size() );
}

//------------------------------------------------------------------------------------------------------------

// Smoothed square hinge function
class CGradientBoostingSquaredHinge : public IGradientBoostingLossFunction {
public:
	// IGradientBoostingLossFunction interface methods
	void CalcGradientAndHessian( const CArray< CArray<double> >& predicts, const CArray< CArray<double> >& answers,
		CArray< CArray<double> >& gradient, CArray< CArray<double> >& hessian ) const override;

	double CalcLossMean( const CArray< CArray<double> >& predicts, const CArray< CArray<double> >& answers ) const override;
};

void CGradientBoostingSquaredHinge::CalcGradientAndHessian( const CArray< CArray<double> >& predicts,
	const CArray< CArray<double> >& answers, CArray< CArray<double> >& gradients, CArray< CArray<double> >& hessians ) const
{
	NeoAssert( predicts.Size() == answers.Size() );

	gradients.SetSize( predicts.Size() );
	hessians.SetSize( predicts.Size() );

	for( int i = 0; i < predicts.Size(); i++ ) {
		gradients[i].SetSize( predicts[i].Size() );
		hessians[i].SetSize( predicts[i].Size() );
		for( int j = 0; j < predicts[i].Size(); j++ ) {
			const double t = -( 2 * answers[i][j] - 1 );

			if( t * predicts[i][j] < 1 ) {
				gradients[i][j] = static_cast<double>( 2 * t * ( t * predicts[i][j] - 1 ) );
				hessians[i][j] = static_cast<double>( 2 * t * t );
			} else {
				gradients[i][j] = 0.0;
				hessians[i][j] = 1e-16;
			}
		}
	}
}

double CGradientBoostingSquaredHinge::CalcLossMean( const CArray< CArray<double> >& predicts,
	const CArray< CArray<double> >& answers ) const
{
	NeoAssert( predicts.Size() == answers.Size() );

	double overallSum = 0;
	auto getMean = []( double sum, int n ) { return n != 0 ? sum / static_cast<double>( n ) : 0; };
	for( int i = 0; i < predicts.Size(); ++i ) {
		double sum = 0;
		for( int j = 0; j < predicts[i].Size(); ++j ) {
			const double base = max( 0.0, 1.0 - ( 2.0 * answers[i][j] - 1.0 ) * predicts[i][j] );
			sum += base * base;
		}
		overallSum += getMean( sum, predicts[i].Size() );
	}

	return getMean( overallSum, predicts.Size() );
}

//------------------------------------------------------------------------------------------------------------

// Quadratic loss function for classification and regression
class CGradientBoostingSquareLoss : public IGradientBoostingLossFunction {
public:
	// IGradientBoostingLossFunction interface methods
	void CalcGradientAndHessian( const CArray< CArray<double> >& predicts, const CArray< CArray<double> >& answers,
		CArray< CArray<double> >& gradient, CArray< CArray<double> >& hessian ) const override;

	double CalcLossMean( const CArray< CArray<double> >& predicts, const CArray< CArray<double> >& answers ) const override;
};

void CGradientBoostingSquareLoss::CalcGradientAndHessian( const CArray< CArray<double> >& predicts,
	const CArray< CArray<double> >& answers, CArray< CArray<double> >& gradients, CArray< CArray<double> >& hessians ) const
{
	NeoAssert( predicts.Size() == answers.Size() );

	gradients.SetSize( predicts.Size() );
	hessians.SetSize( predicts.Size() );

	for( int i = 0; i < predicts.Size(); i++ ) {
		gradients[i].SetSize( predicts[i].Size() );
		hessians[i].SetSize( predicts[i].Size() );
		for( int j = 0; j < predicts[i].Size(); j++ ) {
			gradients[i][j] = static_cast<double>( predicts[i][j] - answers[i][j] );
			hessians[i][j] = static_cast<double>( 1.0 );
		}
	}
}

double CGradientBoostingSquareLoss::CalcLossMean( const CArray< CArray<double> >& predicts,
	const CArray< CArray<double> >& answers ) const
{
	NeoAssert( predicts.Size() == answers.Size() );

	double overallSum = 0;
	auto getMean = []( double sum, int n ) { return n != 0 ? sum / static_cast<double>( n ) : 0; };
	for( int i = 0; i < predicts.Size(); ++i ) {
		double sum = 0;
		for( int j = 0; j < predicts[i].Size(); ++j ) {
			const double diff = answers[i][j] - predicts[i][j];
			sum += diff * diff / 2.0;
		}
		overallSum += getMean( sum, predicts[i].Size() );
	}

	return getMean( overallSum, predicts.Size() );
}

//-------------------------------------------------------------------------------------------------------------

namespace {

// Abstract base class
struct IGBoostPredictionsThreadTask : public IGradientBoostThreadTask {
protected:
	// Create a task
	IGBoostPredictionsThreadTask( IThreadPool&, const IMultivariateRegressionProblem&,
		const CArray<CGradientBoostEnsemble>& models,
		CArray<CArray<CGradientBoost::CPredictionCacheItem>>& predictCache,
		CArray<CArray<double>>& predicts, CArray<CArray<double>>& answers,
		float learningRate, bool isMultiTreesModel );

	// The number of separate executors
	int ThreadCount() const { return ThreadPool.Size(); }
	// Run on each problem's element separately
	void RunOnElement( int threadIndex, int index, int usedVectorIndex,
		const CFloatVectorDesc&, const CFloatVector& );
	// Run the process in a separate thread
	void Run( int threadIndex, int startIndex, int count ) override final;
	// Contains the mapping of the index in the truncated training set for the given step
	virtual int UsedVectorIndex( int index ) const = 0;

	const IMultivariateRegressionProblem& Problem; //performing problem
	const CFloatMatrixDesc Matrix; //performing problem's sizes
	const CArray<CGradientBoostEnsemble>& Models; //given models for multi-class classification
	CArray<CArray<CGradientBoost::CPredictionCacheItem>>& PredictCache; //cache for predictions
	CArray<CArray<double>>& Predicts; //current algorithm predictions on each step
	CArray<CArray<double>>& Answers; //correct answers for the vectors used on each step
	const float LearningRate; //each classifier's multiplier
	const bool IsMultiTreesModel; //multiple or single number of trees in a model
	const int CurStep; //current step of algorithm
	CArray<CFastArray<double, 1>> Predictions{}; //intermediate result
};

IGBoostPredictionsThreadTask::IGBoostPredictionsThreadTask(
		IThreadPool& threadPool,
		const IMultivariateRegressionProblem& problem,
		const CArray<CGradientBoostEnsemble>& models,
		CArray<CArray<CGradientBoost::CPredictionCacheItem>>& predictCache,
		CArray<CArray<double>>& predicts,
		CArray<CArray<double>>& answers,
		float learningRate,
		bool isMultiTreesModel ) :
	IGradientBoostThreadTask( threadPool ),
	Problem( problem ),
	Matrix( Problem.GetMatrix() ),
	Models( models ),
	PredictCache( predictCache ),
	Predicts( predicts ),
	Answers( answers ),
	LearningRate( learningRate ),
	IsMultiTreesModel( isMultiTreesModel ),
	CurStep( models[0].Size() )
{
	NeoAssert( Matrix.Height == Problem.GetVectorCount() );
	NeoAssert( Matrix.Width == Problem.GetFeatureCount() );

	Predictions.SetSize( ThreadCount() );
	for( int t = 0; t < Predictions.Size(); ++t ) {
		Predictions[t].SetSize( Problem.GetValueSize() );
	}
}

void IGBoostPredictionsThreadTask::Run( int threadIndex, int startIndex, int count )
{
	const int endIndex = startIndex + count;
	for( int index = startIndex; index < endIndex; ++index ) {
		const int usedVectorIndex = UsedVectorIndex( index );
		const CFloatVector value = Problem.GetValue( usedVectorIndex );
		CFloatVectorDesc desc;
		Matrix.GetRow( usedVectorIndex, desc );
		// main function call
		RunOnElement( threadIndex, index, usedVectorIndex, desc, value );
	}
}

void IGBoostPredictionsThreadTask::RunOnElement( int threadIndex, int index, int usedVectorIndex,
	const CFloatVectorDesc& desc, const CFloatVector& value )
{
	if( IsMultiTreesModel ) {
		CGradientBoostModel::PredictRaw( Models[0], PredictCache[0][usedVectorIndex].Step,
			 LearningRate, desc, Predictions[threadIndex] );
	} else {
		CFastArray<double, 1> pred{};
		pred.SetSize( 1 );
		for( int j = 0; j < Problem.GetValueSize(); ++j ) {
			CGradientBoostModel::PredictRaw( Models[j], PredictCache[j][usedVectorIndex].Step,
				LearningRate, desc, pred );
			Predictions[threadIndex][j] = pred[0];
		}
	}

	for( int j = 0; j < Problem.GetValueSize(); ++j ) {
		PredictCache[j][usedVectorIndex].Value += Predictions[threadIndex][j];
		PredictCache[j][usedVectorIndex].Step = CurStep;
		Predicts[j][index] = PredictCache[j][usedVectorIndex].Value;
		Answers[j][index] = value[j];
	}
}

//------------------------------------------------------------------------------------------------------------

// Builds the ensemble predictions for a set of vectors
struct CGBoostBuildPredictionsThreadTask : public IGBoostPredictionsThreadTask {
	// Create a task
	CGBoostBuildPredictionsThreadTask(
			IThreadPool& threadPool,
			const IMultivariateRegressionProblem& problem,
			const CArray<CGradientBoostEnsemble>& models,
			CArray<CArray<CGradientBoost::CPredictionCacheItem>>& predictCache,
			CArray<CArray<double>>& predicts,
			CArray<CArray<double>>& answers,
			const CArray<int>& usedVectors,
			float learningRate,
			bool isMultiTreesModel ) :
		IGBoostPredictionsThreadTask( threadPool, problem, models,
			predictCache, predicts, answers, learningRate, isMultiTreesModel ),
		UsedVectors( usedVectors )
	{}
protected:
	int ParallelizeSize() const override { return UsedVectors.Size(); }
	int UsedVectorIndex( int index ) const override { return UsedVectors[index]; }

	const CArray<int>& UsedVectors;
};

//------------------------------------------------------------------------------------------------------------

// Fills the prediction cache with the values of the full problem
struct CGBoostBuildFullPredictionsThreadTask : public IGBoostPredictionsThreadTask {
	// Create a task
	CGBoostBuildFullPredictionsThreadTask( IThreadPool&, const IMultivariateRegressionProblem&,
			const CArray<CGradientBoostEnsemble>& models,
			CArray<CArray<CGradientBoost::CPredictionCacheItem>>& predictCache,
			CArray<CArray<double>>& predicts, CArray<CArray<double>>& answers,
			float learningRate, bool isMultiTreesModel );
protected:
	int ParallelizeSize() const override { return Problem.GetVectorCount(); }
	int UsedVectorIndex( int index ) const override { return index; }
};

CGBoostBuildFullPredictionsThreadTask::CGBoostBuildFullPredictionsThreadTask(
		IThreadPool& threadPool,
		const IMultivariateRegressionProblem& problem,
		const CArray<CGradientBoostEnsemble>& models,
		CArray<CArray<CGradientBoost::CPredictionCacheItem>>& predictCache,
		CArray<CArray<double>>& predicts,
		CArray<CArray<double>>& answers,
		float learningRate,
		bool isMultiTreesModel ) :
	IGBoostPredictionsThreadTask( threadPool, problem, models,
		predictCache, predicts, answers, learningRate, isMultiTreesModel )
{
	for( int i = 0; i < Predicts.Size(); ++i ) {
		Predicts[i].SetSize( Problem.GetVectorCount() );
		Answers[i].SetSize( Problem.GetVectorCount() );
	}
}

} // namespace

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
	result.QuickSort<Ascending<int>>();
}

//------------------------------------------------------------------------------------------------------------

CGradientBoost::CGradientBoost( const CParams& _params ) :
	threadPool( CreateThreadPool( _params.ThreadCount ) ),
	params( _params, threadPool->Size() )
{
	NeoAssert( threadPool != nullptr );
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
	delete threadPool;
}

CPtr<IMultivariateRegressionModel> CGradientBoost::TrainRegression(
	const IMultivariateRegressionProblem& problem )
{
	while( !TrainStep( problem ) ) {};
	return GetMultivariateRegressionModel( problem );
}

CPtr<IRegressionModel> CGradientBoost::TrainRegression( const IRegressionProblem& problem )
{
	while( !TrainStep( problem ) ) {};
	return GetRegressionModel( problem );
}

CPtr<IModel> CGradientBoost::Train( const IProblem& problem )
{
	while( !TrainStep( problem ) ) {};
	return GetClassificationModel( problem );
}

// Creates a tree builder depending on the problem type
void CGradientBoost::createTreeBuilder( const IMultivariateRegressionProblem* problem )
{
	switch( params.TreeBuilder ) {
		case GBTB_Full:
		case GBTB_MultiFull:
		{
			CGradientBoostFullTreeBuilderParams builderParams;
			builderParams.L1RegFactor = params.L1RegFactor;
			builderParams.L2RegFactor = params.L2RegFactor;
			builderParams.MinSubsetHessian = 1e-3f;
			builderParams.ThreadCount = params.ThreadCount;
			builderParams.MaxTreeDepth = params.MaxTreeDepth;
			builderParams.MaxNodesCount = params.MaxNodesCount;
			builderParams.PruneCriterionValue = params.PruneCriterionValue;
			builderParams.MinSubsetWeight = params.MinSubsetWeight;
			builderParams.DenseTreeBoostCoefficient = params.DenseTreeBoostCoefficient;
			if( params.TreeBuilder == GBTB_MultiFull ) {
				fullMultiClassTreeBuilder = FINE_DEBUG_NEW
					CGradientBoostFullTreeBuilder<CGradientBoostStatisticsMulti>( builderParams, logStream );
			} else {
				fullSingleClassTreeBuilder = FINE_DEBUG_NEW
					CGradientBoostFullTreeBuilder<CGradientBoostStatisticsSingle>( builderParams, logStream );
			}
			fullProblem = FINE_DEBUG_NEW CGradientBoostFullProblem( params.ThreadCount, problem,
				usedVectors, usedFeatures, featureNumbers );
			break;
		}
		case GBTB_FastHist:
		case GBTB_MultiFastHist:
		{
			CGradientBoostFastHistTreeBuilderParams builderParams;
			builderParams.L1RegFactor = params.L1RegFactor;
			builderParams.L2RegFactor = params.L2RegFactor;
			builderParams.MinSubsetHessian = 1e-3f;
			builderParams.ThreadCount = params.ThreadCount;
			builderParams.MaxTreeDepth = params.MaxTreeDepth;
			builderParams.MaxNodesCount = params.MaxNodesCount;
			builderParams.PruneCriterionValue = params.PruneCriterionValue;
			builderParams.MaxBins = params.MaxBins;
			builderParams.MinSubsetWeight = params.MinSubsetWeight;
			builderParams.DenseTreeBoostCoefficient = params.DenseTreeBoostCoefficient;
			if( params.TreeBuilder == GBTB_MultiFastHist ) {
				fastHistMultiClassTreeBuilder = FINE_DEBUG_NEW
					CGradientBoostFastHistTreeBuilder<CGradientBoostStatisticsMulti>(
						builderParams, logStream, problem->GetValueSize() );
			} else {
				fastHistSingleClassTreeBuilder = FINE_DEBUG_NEW
					CGradientBoostFastHistTreeBuilder<CGradientBoostStatisticsSingle>( builderParams, logStream, 1 );
			}
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
	fullSingleClassTreeBuilder.Release();
	fullMultiClassTreeBuilder.Release();
	fullProblem.Release();
	fastHistSingleClassTreeBuilder.Release();
	fastHistMultiClassTreeBuilder.Release();
	fastHistProblem.Release();
	baseProblem.Release();
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
void CGradientBoost::initialize()
{
	const int modelCount = baseProblem->GetValueSize();
	const int vectorCount = baseProblem->GetVectorCount();
	const int featureCount = baseProblem->GetFeatureCount();

	NeoAssert( modelCount >= 1 );
	NeoAssert( vectorCount > 0 );
	NeoAssert( featureCount > 0 );

	lossFunction = createLossFunction();
	models.SetSize( isMultiTreesModel() ? 1 : modelCount );

	if( predictCache.Size() == 0 ) {
		predictCache.SetSize( modelCount );
		CPredictionCacheItem item;
		item.Step = 0;
		item.Value = 0;
		for( int i = 0; i < predictCache.Size(); i++ ) {
			predictCache[i].Add( item, vectorCount );
		}
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
		for( int i = 0; i < featureCount; i++ ) {
			usedFeatures.Add( i );
			featureNumbers.Add( i );
		}
	}

	try {
		createTreeBuilder( baseProblem );
	} catch( ... ) {
		destroyTreeBuilder(); // return to the initial state
		throw;
	}

	if( fullProblem != nullptr && params.Subfeature == 1.0 && params.Subsample == 1.0 ) {
		fullProblem->Update();
	}
}

// Performs gradient boosting iteration
// On a sub-problem of the first problem using cache
void CGradientBoost::executeStep( IGradientBoostingLossFunction& lossFunction,
	const IMultivariateRegressionProblem* problem, CGradientBoostEnsemble& curModels )
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

	for( int i = 0; i < predicts.Size(); i++ ) {
		predicts[i].SetSize( usedVectors.Size() );
		answers[i].SetSize( usedVectors.Size() );
		gradients[i].Empty();
		hessians[i].Empty();
	}

	// Build the current model predictions
	CGBoostBuildPredictionsThreadTask( *threadPool, *problem, models,
		predictCache, predicts, answers, usedVectors,
		params.LearningRate, isMultiTreesModel() ).ParallelRun();

	// The vectors in the regression value are partial derivatives of the loss function
	// The tree built for this problem will decrease the loss function value
	lossFunction.CalcGradientAndHessian( predicts, answers, gradients, hessians );

	// Add the vector weights and calculate the total
	CArray<double> gradientsSum;
	gradientsSum.Add( 0, gradients.Size() );
	CArray<double> hessiansSum;
	hessiansSum.Add( 0, gradients.Size() );
	CArray<double> weights;
	weights.SetSize( usedVectors.Size() );

	double weightsSum = 0;
	for( int i = 0; i < usedVectors.Size(); i++ ) {
		weights[i] = problem->GetVectorWeight( usedVectors[i] );
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

	if( params.Subfeature != 1.0 || params.Subsample != 1.0 ) {
		// The sub-problem data has changed, reload it
		if( fullProblem != nullptr ) {
			fullProblem->Update();
		}
	}

	if( fullMultiClassTreeBuilder != nullptr || fastHistMultiClassTreeBuilder != nullptr ) {
		if( fullMultiClassTreeBuilder != nullptr ) {
			curModels.Add( fullMultiClassTreeBuilder->Build( *fullProblem,
				gradients, gradientsSum, hessians, hessiansSum, weights, weightsSum ).Ptr() );
		} else {
			curModels.Add( fastHistMultiClassTreeBuilder->Build( *fastHistProblem, gradients, hessians, weights ).Ptr() );
		}
	} else {
		for( int i = 0; i < gradients.Size(); i++ ) {
			if( logStream != nullptr ) {
				*logStream << "GradientSum = " << gradientsSum[i]
					<< " HessianSum = " << hessiansSum[i]
					<< "\n";
			}
			CPtr<IRegressionTreeNode> model;
			if( fullSingleClassTreeBuilder != nullptr ) {
				model = fullSingleClassTreeBuilder->Build( *fullProblem,
					gradients[i], gradientsSum[i],
					hessians[i], hessiansSum[i],
					weights, weightsSum );
			} else {
				model = fastHistSingleClassTreeBuilder->Build( *fastHistProblem, gradients[i], hessians[i], weights );
			}
			curModels.Add( model );
		}
	}
}

// Creates model represetation requested in params.
CPtr<IObject> CGradientBoost::createOutputRepresentation(
	CArray<CGradientBoostEnsemble>& models, int predictionSize )
{
	CPtr<CGradientBoostModel> linked = FINE_DEBUG_NEW CGradientBoostModel(
		models, predictionSize, params.LearningRate, params.LossFunction );

	switch( params.Representation ) {
		case GBMR_Linked:
			return linked.Ptr();
		case GBMR_Compact:
			linked->ConvertToCompact();
			return linked.Ptr();
		case GBMR_QuickScorer:
			return CGradientBoostQuickScorer().Build( *linked ).Ptr();
		default:
			NeoAssert( false );
			return 0;
	}
}

void CGradientBoost::prepareProblem( const IProblem& _problem )
{
	if( baseProblem == 0 ) {
		CPtr<const IMultivariateRegressionProblem> multivariate;
		if( _problem.GetClassCount() == 2 ) {
			multivariate = FINE_DEBUG_NEW CMultivariateRegressionOverBinaryClassification( &_problem );
		} else {
			multivariate = FINE_DEBUG_NEW CMultivariateRegressionOverClassification( &_problem );
		}

		baseProblem = FINE_DEBUG_NEW CMultivariateRegressionProblemNotNullWeightsView( multivariate );
		initialize();
	}
}

void CGradientBoost::prepareProblem( const IRegressionProblem& _problem )
{
	if( baseProblem == 0 ) {
		CPtr<const IMultivariateRegressionProblem> multivariate =
			FINE_DEBUG_NEW CMultivariateRegressionOverUnivariate( &_problem );
		baseProblem = FINE_DEBUG_NEW CMultivariateRegressionProblemNotNullWeightsView( multivariate );
		initialize();
	}
}

void CGradientBoost::prepareProblem( const IMultivariateRegressionProblem& _problem )
{
	if( baseProblem == 0 ) {
		baseProblem = FINE_DEBUG_NEW CMultivariateRegressionProblemNotNullWeightsView( &_problem );
		initialize();
	}
}

bool CGradientBoost::TrainStep( const IProblem& _problem )
{
	prepareProblem( _problem );
	return trainStep();
}

bool CGradientBoost::TrainStep( const IRegressionProblem& _problem )
{
	prepareProblem( _problem );
	return trainStep();
}

bool CGradientBoost::TrainStep( const IMultivariateRegressionProblem& _problem )
{
	prepareProblem( _problem );
	return trainStep();
}

bool CGradientBoost::trainStep()
{
	try {
		if( logStream != nullptr ) {
			*logStream << "\nBoost iteration " << models[0].Size() << ":\n";
		}

		// Gradient boosting step
		CGradientBoostEnsemble curIterationModels; // a new model for multi-class classification
		executeStep( *lossFunction, baseProblem, curIterationModels );

		for( int j = 0; j < curIterationModels.Size(); j++ ) {
			models[j].Add( curIterationModels[j] );
		}
	} catch( ... ) {
		destroyTreeBuilder(); // return to the initial state
		throw;
	}

	return models[0].Size() >= params.IterationsCount;
}

void CGradientBoost::Serialize( CArchive& archive )
{
	if( archive.IsStoring() ) {
		archive << models.Size();
		if( models.Size() > 0 ) {
			archive << models[0].Size();
			for( int i = 0; i < models.Size(); i++ ) {
				CGradientBoostEnsemble& ensemble = models[i];
				for( int j = 0; j < ensemble.Size(); j++ ) {
					ensemble[j]->Serialize( archive );
				}
			}
		}
		predictCache.Serialize( archive );
	} else {
		int ensemblesCount;
		archive >> ensemblesCount;
		if( ensemblesCount > 0 ) {
			models.SetSize( ensemblesCount );
			int iterationsCount;
			archive >> iterationsCount;
			if( iterationsCount > 0 ) {
				for( int i = 0; i < models.Size(); i++ ) {
					models[i].SetSize( iterationsCount );
					for( int j = 0; j < iterationsCount; j++ ) {
						models[i][j] = CreateModel<IRegressionTreeNode>( "FmlRegressionTreeModel" );
						models[i][j]->Serialize( archive );
					}
				}
			}
		}
		predictCache.Serialize( archive );
	}
}

template<typename T>
CPtr<T> CGradientBoost::getModel()
{
	// Calculate the last loss values
	CGBoostBuildFullPredictionsThreadTask( *threadPool, *baseProblem, models,
		predictCache, predicts, answers, params.LearningRate, isMultiTreesModel() ).ParallelRun();
	loss = lossFunction->CalcLossMean( predicts, answers );

	int predictionSize = isMultiTreesModel() ? baseProblem->GetValueSize() : 1;
	destroyTreeBuilder();
	predictCache.DeleteAll();

	return CheckCast<T>( createOutputRepresentation( models, predictionSize ) );
}

CPtr<IModel> CGradientBoost::GetClassificationModel( const IProblem& _problem )
{
	prepareProblem( _problem );
	return getModel<IModel>();
}

CPtr<IRegressionModel> CGradientBoost::GetRegressionModel( const IRegressionProblem& _problem )
{
	prepareProblem( _problem );
	return getModel<IRegressionModel>();
}

CPtr<IMultivariateRegressionModel> CGradientBoost::GetMultivariateRegressionModel(
	const IMultivariateRegressionProblem& _problem )
{
	prepareProblem( _problem );
	return getModel<IMultivariateRegressionModel>();
}

} // namespace NeoML
