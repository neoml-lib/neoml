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

#include <NeoML/TraditionalML/EMClustering.h>
#include <float.h>

namespace NeoML {

static const double MaxExp = 700; // the maximum acceptable exponent value

CEMClustering::CParam::CParam() :
	RoughInitCount( 3 ),
	FinalInitCount( 1 ),
	MaxFixedEmIteration( 10 ),
	ConvThreshold( 0.5 ),
	InitialClustersCount( 2 ),
	MaxClustersCount( 2 ),
	MinClusterSize( 1 )
{
}

CEMClustering::CParam::CParam( const CParam& result ) :
	RoughInitCount( result.RoughInitCount ),
	FinalInitCount( result.FinalInitCount ),
	MaxFixedEmIteration( result.MaxFixedEmIteration ),
	ConvThreshold( result.ConvThreshold ),
	InitialClustersCount( result.InitialClustersCount ),
	MaxClustersCount( result.MaxClustersCount ),
	MinClusterSize( result.MinClusterSize )
{
	result.InitialClusters.CopyTo( InitialClusters );
}

//---------------------------------------------------------------------------------------------------------

CEMClustering::CEmClusteringResult::CEmClusteringResult() :
	Likelihood( 0 ),
	Bic( 0 ),
	Aic( 0 ),
	IsGood( false )
{
}

CEMClustering::CEmClusteringResult::CEmClusteringResult( const CEmClusteringResult& result ) :
	Likelihood( result.Likelihood ),
	Bic( result.Bic ),
	Aic( result.Aic ),
	IsGood( result.IsGood )
{
	result.Result.CopyTo( Result );
}

CEMClustering::CEmClusteringResult& CEMClustering::CEmClusteringResult::operator=( const CEMClustering::CEmClusteringResult& other )
{
	Likelihood = other.Likelihood;
	Bic = other.Bic;
	Aic = other.Aic;
	IsGood = other.IsGood;
	other.Result.CopyTo( Result );
	return *this;
}

//---------------------------------------------------------------------------------------------------------

CEMClustering::CEMClustering( const CParam& _params ) :
	params( _params ),
	log( 0 )
{
}

CEMClustering::~CEMClustering()
{
}

bool CEMClustering::Clusterize( IClusteringData* data, CClusteringResult& result )
{
	NeoAssert( data != 0 );

	if( log != 0 ) {
		*log << "\nEM clustering started:\n";
	}

	CSparseFloatMatrixDesc matrix = data->GetMatrix();
	NeoAssert( matrix.Height == data->GetVectorCount() );
	NeoAssert( matrix.Width == data->GetFeaturesCount() );

	CArray<double> weights;
	for( int i = 0; i < matrix.Height; i++ ) {
		weights.Add( data->GetVectorWeight( i ) );
	}

	for( int k = params.InitialClustersCount; k <= params.MaxClustersCount; k++ ) {
		if( log != 0 ) {
			*log << "\n[Step " << k - params.InitialClustersCount << "]\n";
		}

		int iterationsCount = params.RoughInitCount;
		if( k == 1 ) {
			// One run is enough because the optimization problem can be solved exactly
			iterationsCount = 1;
		}
		CEmClusteringResult emResult;
		runEMFixedComponents( matrix, weights, k, iterationsCount, true, emResult );

		if( !emResult.IsGood ) {
			// No "good" results found, looking for not necessarily "good"
			runEMFixedComponents( matrix, weights, k, iterationsCount, false, emResult );
		}

		history.Add( emResult );
		if( log != 0 ) {
			*log << "Aic: " << emResult.Aic << "\n";
			*log << "Bic: " << emResult.Bic << "\n";
			*log << "Likelihood: " << emResult.Likelihood << "\n";
			*log << "IsGood: " << emResult.IsGood << "\n";
			*log << "ClusterCount: " << emResult.Result.ClusterCount << "\n";
			for( int i = 0; i < emResult.Result.ClusterCount; i++ ) {
				*log << "Cluster " << i << ":\n";
				*log << emResult.Result.Clusters[i];
			}
		}
	}

	CEmClusteringResult emBestResult;
	findBestResult( matrix, weights, emBestResult );
	emBestResult.Result.CopyTo( result );
	return emBestResult.IsGood;
}

// Runs EM clustering with the fixed number of clusters
void CEMClustering::runEMFixedComponents( const CSparseFloatMatrixDesc& data, const CArray<double>& weights, int clustersCount, int iterationsCount, bool goodOnly,
	CEmClusteringResult& bestResult )
{
	CArray<CClusterCenter> initialClusters;
	calculateInitialClusters( data, clustersCount, initialClusters );
	
	bestResult.Likelihood = -DBL_MAX;
	CEmClusteringResult iterationResult;

	for( int i = 0; i < iterationsCount; i++ ) {
		clusterize( data, weights, initialClusters, iterationResult );

		if( !goodOnly || iterationResult.IsGood ) {
			if( iterationResult.Likelihood > bestResult.Likelihood ) {
				bestResult = iterationResult;
			}
		}

		recalculateInitialClusters( data, iterationResult, initialClusters );
	}
}

// Selects a random cluster
int CEMClustering::selectRandomCluster( const CFastArray<double, 1>& cumulativeFitnesses ) const
{
	// A cluster may come up with the probability in inverse proportion to its size
	const double randDouble = static_cast<double>( rand() ) / static_cast<double>( RAND_MAX );
	for( int i = 0; i < cumulativeFitnesses.Size(); i++ ) {
		if( cumulativeFitnesses[i] >= randDouble ) {
			return i;
		}
	}

	return cumulativeFitnesses.Size() - 1;
}

// Finds the best result
void CEMClustering::findBestResult( const CSparseFloatMatrixDesc& data, const CArray<double>& weights, CEmClusteringResult& result )
{
	NeoAssert( !history.IsEmpty() );

	if( log != 0 ) {
		*log << "\nFind best result in history:\n\n";
	}

	int res = 0;
	for( int i = 0; i < history.Size(); i++ ) {
		if( history[i].Bic < history[res].Bic ) {
			res = i;
		}
	}
	const int bestClustersCount = history[res].Result.Clusters.Size();
	
	CEmClusteringResult bestClustersCountResult;
	runEMFixedComponents( data, weights, bestClustersCount, params.FinalInitCount, true, bestClustersCountResult );

	const bool isFinalLikelihoodBetter = bestClustersCountResult.Likelihood > history[res].Likelihood;

	if( bestClustersCountResult.IsGood > history[res].IsGood ||
		( bestClustersCountResult.IsGood == history[res].IsGood && isFinalLikelihoodBetter ) )
	{
		result = bestClustersCountResult;
	} else {
		result = history[res];
	}
}

// Initializes the starting clusters
void CEMClustering::calculateInitialClusters( const CSparseFloatMatrixDesc& data, int clustersCount, CArray<CClusterCenter>& initialClusters ) const
{
	if( params.InitialClusters.Size() == clustersCount ) {
		params.InitialClusters.CopyTo( initialClusters );
		return; // in this case, the cluster centers were set directly
	}

	// Initialize the cluster centers with a random selection of input data
	const int vectorsCount = data.Height;
	initialClusters.SetSize( clustersCount );
	for( int i = 0; i < clustersCount; i++ ) {
		CSparseFloatVectorDesc desc;
		data.GetRow( rand() % vectorsCount, desc );
		CFloatVector mean( data.Width, desc );
		CClusterCenter center( mean );
		center.Weight = 1.0 / clustersCount;
		initialClusters[i] = center;
	}
}

// Recalculates the starting cluster centers
void CEMClustering::recalculateInitialClusters( const CSparseFloatMatrixDesc& data, const CEmClusteringResult& prevResult,
	CArray<CClusterCenter>& initialClusters ) const
{
	initialClusters.SetSize( prevResult.Result.ClusterCount );

	// Used by roulette wheel selection for initialization
	CFastArray<double, 1> cumulativeFitnesses;
	initCumulativeFitnesses( prevResult.Result.Clusters, cumulativeFitnesses );

	CArray<int> dataClusters;
	prevResult.Result.Data.CopyTo( dataClusters );

	CArray<int> clustersElements;
	for( int i = 0; i < prevResult.Result.ClusterCount; i++ ) {
		// Cluster initializer is another cluster from which a random image is taken to initialize the mean
		const int iClusterInitializator = selectRandomCluster( cumulativeFitnesses );
		int initFrom = NotFound;
		
		clustersElements.DeleteAll();
		for( int j = 0; j < prevResult.Result.Data.Size(); j++ ) {
			if( prevResult.Result.Data[j] == iClusterInitializator ) {
				clustersElements.Add( j );
			}
		}

		if( clustersElements.IsEmpty() ) {
			// The initializer is empty so a random element is taken
			initFrom = rand() % data.Height;
		} else {
			const int indexInCluster = rand() % clustersElements.Size();
			initFrom = clustersElements[indexInCluster];
			// Avoiding repetition for small clusters
			dataClusters[initFrom] = NotFound; // this element will not be used again
		}

		CSparseFloatVectorDesc desc;
		data.GetRow( initFrom, desc );
		initialClusters[i].Mean = CFloatVector( initialClusters[i].Mean.Size(), desc );
		initialClusters[i].Disp = CFloatVector( initialClusters[i].Mean.Size(), 1.0 );
		initialClusters[i].Weight = 1.0 / initialClusters.Size();
	}
}

// Initializes the cumulativeFitnesses array
void CEMClustering::initCumulativeFitnesses( const CArray<CClusterCenter>& initialClusters, CFastArray<double, 1>& cumulativeFitnesses ) const
{
	const int clustersCount = initialClusters.Size();

	NeoAssert( clustersCount > 0 );
	cumulativeFitnesses.SetSize( clustersCount );

	double denom = 0;
	for( int i = 0; i < clustersCount; i++ ) {
		NeoAssert( initialClusters[i].Weight > 0 );
		cumulativeFitnesses[i] = 1.0 / initialClusters[i].Weight;
		denom += cumulativeFitnesses[i];
	}
	for( int i = 0; i < clustersCount; i++ ) {
		cumulativeFitnesses[i] /= denom;
	}

	for( int i = 1; i < clustersCount; i++ ) {
		cumulativeFitnesses[i] += cumulativeFitnesses[i - 1];
	}
}

// Performs clustering starting with the specified initial cluster set
void CEMClustering::clusterize( const CSparseFloatMatrixDesc& data, const CArray<double>& weights,
	const CArray<CClusterCenter>& initialClusters, CEmClusteringResult& result )
{
	if( log != 0 ) {
		*log << "\nEM fixed components clustering started:\n";
	}

	initialClusters.CopyTo( clusters );
	const int vectorsCount = data.Height;
	// Fill the tables with zeros
	hiddenVars.DeleteAll();
	hiddenVars.Add( CFloatVector( clusters.Size(), 0.0 ), vectorsCount ); 
	densitiesArgs.DeleteAll();
	densitiesArgs.Add( CFloatVector( clusters.Size(), 0.0 ), vectorsCount ); 
	calculateDensitiesArgs( data );

	double prevLogLikelihood = 0;
	bool isConverged = false;
	for( int iteration = 0; iteration < params.MaxFixedEmIteration; iteration++ ) {
		// E-step
		expectation();
		// M-step
		maximization( data, weights );

		const double logOfMixtureLikelihood = calculateLogOfMixtureLikelihood();

		if( log != 0 ) {
			*log << "\n[Step " << iteration << "]\n";
			for( int i = 0; i < clusters.Size(); i++ ) {
				*log << "Cluster " << i << ": \n";
				*log << clusters[i];
			}
			*log << "Likelihood: " << logOfMixtureLikelihood << "\n";
		}

		if( iteration > 0 && abs( logOfMixtureLikelihood - prevLogLikelihood ) < params.ConvThreshold ) {
			isConverged = true;
			break;
		}
		prevLogLikelihood = logOfMixtureLikelihood;
	}

	calculateResult( data, isConverged, result );

	if( log != 0 ) {
		if( result.IsGood ) {
			*log << "\nSuccessful!\n";
		} else {
			*log << "\nUnsuccessful!\n";
		}
	}
}

// Performs the E-step of the algorithm ("expectation")
void CEMClustering::expectation()
{
	// The E-step consists of recalculating the hidden variables (stored in the hiddenVars matrix)
	for( int i = 0; i < hiddenVars.Size(); i++ ) {
		for( int j = 0; j < hiddenVars[i].Size(); j++ ) {
			// The numerator logarithm
			const double logOfNumerator = densitiesArgs[i][j];
			double denominator = 0.; // denominator divided by numerator
			bool isNull = false; // a very small fraction will be considered equivalent to 0

			for( int k = 0; k < hiddenVars[i].Size(); k++ ) {
				// First calculate a term logarithm, then its exponent
				const double arg = densitiesArgs[i][k] - logOfNumerator;
				if( arg > MaxExp ) {
					// The denominator is too large, set the fraction value to 0
					isNull = true;
					break;
				}
				denominator += ::exp( arg );
			}
			hiddenVars[i].SetAt( j, static_cast<float>( isNull ? 0 : ( 1. / denominator ) ) );
		}
	}
}

// Performs the M-step of the algorithm ("maximization")
void CEMClustering::maximization( const CSparseFloatMatrixDesc& data, const CArray<double>& weights )
{
	// Retrieve all vectors from the collection
	CArray<CFloatVector> vectors;
	CArray<double> vectorWeight;
	double sumWeight = 0;
	vectors.SetBufferSize( data.Height );
	for( int i = 0; i < data.Height; i++ ) {
		CSparseFloatVectorDesc desc;
		data.GetRow( i, desc );
		CFloatVector vector( data.Width, desc );
		vectors.Add( vector );
		vectorWeight.Add( weights[i] );
		sumWeight += vectorWeight.Last();
	}

	// The M-step consists of recalculating the weights, cluster centers and variances
	calculateNewWeights();
	calculateNewMeans( vectors, vectorWeight, sumWeight );
	calculateNewDisps( vectors, vectorWeight, sumWeight );
	// And finally, the cluster densities
	calculateDensitiesArgs( data );
}

// Recalculates the cluster weights
void CEMClustering::calculateNewWeights()
{
	for( int i = 0; i < clusters.Size(); i++ ) {
		double weight = 0;
		for( int j = 0; j < hiddenVars.Size(); j++ ) {
			weight += hiddenVars[j][i];
		}
		weight /= hiddenVars.Size();
		clusters[i].Weight = weight;
	}
}

// Recalculates the cluster centers
void CEMClustering::calculateNewMeans( const CArray<CFloatVector>& vectors, const CArray<double>& vectorsWeight, double sumWeight )
{
	for( int i = 0; i < clusters.Size(); i++ ) {
		// The weight already calculated
		const double weight = clusters[i].Weight;
		NeoAssert( weight > 0 );

		for( int j = 0; j < clusters[i].Mean.Size(); j++ ) {
			// Calculate the mean value for the j-th feature
			double mean = 0;
			for( int k = 0; k < vectors.Size(); k++ ) {
				mean += vectors[k][j] * hiddenVars[k][i] * vectorsWeight[k];
			}
		
			clusters[i].Mean.SetAt( j, static_cast<float>( mean / ( weight * sumWeight ) ) );
		}
	}
}

// Recalculates variance
void CEMClustering::calculateNewDisps(  const CArray<CFloatVector>& vectors, const CArray<double>& vectorsWeight, double sumWeight  )
{
	for( int i = 0; i < clusters.Size(); i++ ) {
		// The weight already calculated
		const double weight = clusters[i].Weight;
		NeoAssert( weight > 0 );

		for( int j = 0; j < clusters[i].Disp.Size(); j++ ) {
			// Calculate the variance for the j-th feature
			double disp = 0;

			for( int k = 0; k < vectors.Size(); k++ ) {
				const double diff = vectors[k][j] - clusters[i].Mean[j];
				disp += diff * diff * hiddenVars[k][i] * vectorsWeight[k];
			}
		
			disp /= weight * sumWeight;
			clusters[i].Disp.SetAt( j, static_cast<float>( max( disp, 0.5 ) ) ); // make sure there are no 0 variance values
		}
	}
}

// Recalculates the logarithm of density multiplied by weight
void CEMClustering::calculateDensitiesArgs( const CSparseFloatMatrixDesc& data )
{
	for( int j = 0; j < clusters.Size(); j++ ) {
		double logOfWeight = ::log( clusters[j].Weight ); // multiply the weight by density, then take logarithm of the result
		// Each cluster has n-dimensional Gaussian density with the center means[j]
		// and the diagonal variance matrix disps[j].
		// Calculate the logarithm of the coefficient used to calculate density
		double logOfDensityCoeff = -0.5 * ::log( 2 * Pi ) * clusters[j].Disp.Size();
		for( int i = 0; i < clusters[j].Disp.Size(); i++ ) {
			NeoAssert( clusters[j].Disp[i] > 0 );
			logOfDensityCoeff -= 0.5 * ::log( clusters[j].Disp[i] );
		}

		for( int i = 0; i < densitiesArgs.Size(); i++ ) {
			// The final result equals (coefficient + distance to the center + weight)
			CSparseFloatVectorDesc desc;
			data.GetRow( i, desc );
			densitiesArgs[i].SetAt( j, static_cast<float>( logOfDensityCoeff - 0.5 * calculateDistance( j, desc ) + logOfWeight ) );
		}
	}
}

// Calculates weighted Euclidean distance from the cluster center to the given vector
double CEMClustering::calculateDistance( int clusterIndex, const CSparseFloatVectorDesc& desc ) const
{
	NeoPresume( clusterIndex >= 0 );
	NeoPresume( clusterIndex < clusters.Size() );

	double res = 0;
	for( int i = 0; i < desc.Size; i++ ) {
		NeoAssert( desc.Indexes[i] <= clusters[clusterIndex].Mean.Size() );
		const double diff = desc.Values[i] - clusters[clusterIndex].Mean[desc.Indexes[i]];
		NeoAssert( clusters[clusterIndex].Disp[desc.Indexes[i]] > 0 );
		res += diff * diff / clusters[clusterIndex].Disp[desc.Indexes[i]];
	}
	return res;
}

// Retrieves clustering result
void CEMClustering::calculateResult( const CSparseFloatMatrixDesc& data, bool isConverged, CEmClusteringResult& result ) const
{
	CArray<int> clustersSize;
	clustersSize.InsertAt( 0, 0, clusters.Size() );

	clusters.CopyTo( result.Result.Clusters );
	result.Result.ClusterCount = clusters.Size();
	// Each element is assigned to the cluster with the maximum probability for it
	result.Result.Data.SetSize( data.Height );

	for( int i = 0; i < hiddenVars.Size(); i++ ) {
		int res = 0;
		double maxHiddenVar = hiddenVars[i][res];
		for( int j = 1; j < hiddenVars[i].Size(); j++ ) {
			if( hiddenVars[i][j] > maxHiddenVar ) {
				res = j;
				maxHiddenVar = hiddenVars[i][j];
			}
		}
		result.Result.Data[i] = res;
		clustersSize[res]++;
	}

	result.Likelihood = calculateLogOfMixtureLikelihood();
	result.IsGood = isConverged;
	for( int i = 0; i < clustersSize.Size(); i++ ) {
		if( clustersSize[i] < params.MinClusterSize ) {
			result.IsGood = false;
			break;
		}
	}

	const int featuresCount = clusters.First().Mean.Size();
	// means + disps + weights - 1 (because the total weight is 1)
	int freeParametersCount = 2 * clusters.Size() * featuresCount + clusters.Size() - 1;
	result.Bic = -2 * result.Likelihood + freeParametersCount * ::log( static_cast<double>( data.Height ) );
	result.Aic = -2 * result.Likelihood + 2 * freeParametersCount;
}

// Calculates the likelihood function for the entire data set
double CEMClustering::calculateLogOfMixtureLikelihood() const
{
	// Calculate the likelihood logarithm
	// Factor out the exponent with the maximum argument to avoid taking logarithm of zero
	double res = 0.;
	for( int i = 0; i < densitiesArgs.Size(); i++ ) {
		double max = densitiesArgs[i][0];
		for( int j = 0; j < densitiesArgs[i].Size(); j++ ) {
			if( max < densitiesArgs[i][j] ) {
				max = densitiesArgs[i][j];
			}
		}
		double sum = 0;
		for( int j = 0; j < densitiesArgs[i].Size(); j++ ) {
			sum += ::exp( densitiesArgs[i][j] - max );
		}

		NeoAssert( sum > 0 );
		res += max + ::log( sum );
	}

	return res;
}

} // namespace NeoML
