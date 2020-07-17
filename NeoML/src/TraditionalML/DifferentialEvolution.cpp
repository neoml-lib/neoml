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

#include <NeoML/TraditionalML/DifferentialEvolution.h>
#include <NeoML/TraditionalML/Shuffler.h>
#include <float.h>

namespace NeoML {

CDifferentialEvolution::CDifferentialEvolution( IFunctionEvaluation& _func, double fluctuation, double cr, int population ) :
	func( _func ),
	log( 0 ),
	population( population ),
	fluctuation( fluctuation ),
	crossProbability( cr ),
	maxGenerationCount( -1 ),
	generationNum( 0 ),
	lastBestGenerationNum( 0 ),
	maxNonGrowingBestValue( -1 )
{
	NeoAssert( fluctuation > 0. && fluctuation < 1. );
	NeoAssert( crossProbability > 0. && crossProbability < 1. );
	NeoAssert( func.NumberOfDimensions() >= 0 );
	// The function params 'p' may be transferred to the next generation as is
	// Otherwise it will be replaced with mutation of 3 other different params ('a','b','c')
	// That requires at least 4 different vectors in population
	NeoAssert( population >= 4 );
}

// Initializes the parameters using random values in the specified range
CFunctionParamVector CDifferentialEvolution::initPoint() const
{
	CFunctionParamVector ret( func.NumberOfDimensions() );
	CArray<CFunctionParam>& retArray = ret.CopyOnWrite();
	for( int i = 0; i < ret.Size(); ++i ) {
		retArray[i] = func.GetParamTraits( i ).GenerateRandom( random, func.GetMinConstraint( i ), func.GetMaxConstraint( i ) );
	}

	return ret;
}

void CDifferentialEvolution::SetFirstGeneration( const CArray<CFunctionParamVector>& generation )
{
	NeoAssert( generation.Size() <= population );
	curPopulation.SetSize( generation.Size() );
	nextPopulation.SetSize( generation.Size() );
	for( int i = 0; i < generation.Size(); i++ ) {
		curPopulation[i] = generation[i];
		nextPopulation[i] = generation[i];
	}
}

void CDifferentialEvolution::SetFirstGeneration( const CArray<CFunctionParamVector>& generation,
	const CArray<CFunctionParam>& generationValues )
{
	NeoAssert( generation.Size() == generationValues.Size() );
	SetFirstGeneration( generation );
	generationValues.CopyTo( funcValues );
}

// The "p" mutation function with randomly selected parameters: "a", "b", "c"
inline CFunctionParam CDifferentialEvolution::mutate( const IParamTraits& traits, const CFunctionParam& p,
	const CFunctionParam& a, const CFunctionParam& b, const CFunctionParam& c,
	const CFunctionParam& minVal, const CFunctionParam& maxVal )
{
	if( random.Uniform( 0, 1 ) < crossProbability ) {
		return traits.Mutate( random, c, a, b, fluctuation, minVal, maxVal );
	} else {
		return p;
	}
}

void CDifferentialEvolution::initializeAlgo()
{
	// Initialize the starting and the next generation using random values
	// unless they are already initialized by a direct call to SetFirstGeneration
	int initialSize = curPopulation.Size();
	if( initialSize < population ) {
		curPopulation.Grow( population );
		nextPopulation.Grow( population );
		for( int p = initialSize; p < population; ++p ) {
			curPopulation.Add( initPoint() );
			nextPopulation.Add( initPoint() );
		}
	}
	// Calculate the quality of the initial generation
	if( funcValues.Size() == 0 ) {
		func.Evaluate( curPopulation, funcValues );
	}
	NeoAssert( funcValues.Size() == curPopulation.Size() );

	// Looking for the best value
	const IParamTraits& traits = func.GetResultTraits();
	for( int i = 0; i < funcValues.Size(); ++i ) {
		if( i == 0 || traits.Less( funcValues[i], lastBestValue ) ) {
			lastBestValue = funcValues[i];
		}
	}
	lastBestGenerationNum = 0;
}

bool CDifferentialEvolution::BuildNextGeneration()
{
	if( generationNum == 0 ) {
		initializeAlgo();
	}
	generationNum += 1;

	////////////// Mutate the current generation ////////////////
	CArray<CFunctionParamVector> trials; // trials[i] is the mutation of curPopulation[i]
	trials.SetBufferSize( curPopulation.Size() );
	for( int p = 0; p < curPopulation.Size(); ++p ) {
		// Choose three random elements from the current generation
		CShuffler shuffler( random, curPopulation.Size() );
		shuffler.SetNext( p );

		int a = shuffler.Next();
		NeoAssert( a >= 0 && a < curPopulation.Size() );

		int b = shuffler.Next();
		NeoAssert( b >= 0 && b < curPopulation.Size() );

		int c = shuffler.Next();
		NeoAssert( c >= 0 && c < curPopulation.Size() );

		// Add the mutated parameters vector
		CFunctionParamVector trialAdd( func.NumberOfDimensions() );
		CArray<CFunctionParam>& trialAddArr = trialAdd.CopyOnWrite();
		for( int i = 0; i < trialAdd.Size(); ++i ) {
			trialAddArr[i] = mutate( func.GetParamTraits( i ), curPopulation[p][i],
				curPopulation[a][i], curPopulation[b][i], curPopulation[c][i],
				func.GetMinConstraint( i ), func.GetMaxConstraint( i ) );
		}
		trials.Add( trialAdd );
	}

	////////////// Evaluate the function on the mutated elements ////////////////
	CArray<CFunctionParam> trialFuncValues;
	func.Evaluate( trials, trialFuncValues );

	////////////// Create the next generation ////////////////
	const IParamTraits& traits = func.GetResultTraits();
	for( int p = 0; p < curPopulation.Size(); ++p ) {
		// If the function value improved on this parameter set, remember it
		if( traits.Less( trialFuncValues[p], funcValues[p] ) ) {
			funcValues[p] = trialFuncValues[p];
			nextPopulation[p] = trials[p];

			if( traits.Less( trialFuncValues[p], lastBestValue ) ) {
				lastBestValue = trialFuncValues[p];
				lastBestGenerationNum = generationNum;
			}
		} else { // Otherwise keep the previous generation element
			nextPopulation[p] = curPopulation[p];
		}
	}
	//////////////// Generation change /////////////////
	// Writing the generation
	for( int p = 0; p < curPopulation.Size(); ++p ) {
		curPopulation[p] = nextPopulation[p];
	}

	return checkStop();
}

bool CDifferentialEvolution::checkStop()
{
	// The maximum number of generations is reached
	if( maxGenerationCount >= 0 && generationNum >= maxGenerationCount ) {
		if( log != 0 ) {
			*log << "DiffEvolution: Max Generation Count reached- " << maxGenerationCount << "\n";
		}
		return true;
	}

	// Check that the best value has not improved for a long time
	if( maxNonGrowingBestValue >= 0 && generationNum - lastBestGenerationNum > maxNonGrowingBestValue ) {
		if( log != 0 ) {
			*log << "DiffEvolution: best value not growing - " <<
				generationNum << " - " << lastBestGenerationNum << " > " << maxNonGrowingBestValue << "\n";
		}
		return true;
	}

	// Check if the data has degenerated
	bool extinction = true;
	int dims = func.NumberOfDimensions();
	for( int p = 1; extinction && p < curPopulation.Size(); ++p ) {
		for( int i = 0; extinction && i < dims; ++i ) {
			const IParamTraits& paramTraits = func.GetParamTraits( i );
			if( paramTraits.Less( curPopulation[p][i], curPopulation[0][i] ) ||
				paramTraits.Less( curPopulation[0][i], curPopulation[p][i] ) ) {
				extinction = false;
			}
		}
	}
	if( extinction ) {
		if( log != 0 ) {
			*log << "DiffEvolution: extinction\n";
		}
		return true;
	}

	return false;
}

CFunctionParamVector CDifferentialEvolution::GetOptimalVector() const
{
	int resultIndex = 0;
	CFunctionParam value;
	const IParamTraits& traits = func.GetResultTraits();
	for( int i = 0; i < curPopulation.Size(); ++i ) {
		if( i == 0 || traits.Less( funcValues[i], value ) ) {
			value = funcValues[i];
			resultIndex = i;
		}
	}

	return curPopulation[resultIndex];
}

void CDifferentialEvolution::LogPopulation()
{
	if( log == 0 ) {
		return;
	}

	*log << ">>>>>>>>DiffEvolution>>>>>>>>--------\n";
	*log << "Generation " << generationNum << "\n";
	for( int i = 0; i < curPopulation.Size(); ++i ) {
		*log << i << " - FuncValue ";
		*log << func.GetResultTraits();
		IParamTraits::CDumper resultDumper( *log, func.GetResultTraits() );
		resultDumper << funcValues[i];
		*log << " vector";
		for( int j = 0; j < curPopulation[i].Size(); ++j ) {
			*log << " ";
			*log << func.GetParamTraits( j );
			IParamTraits::CDumper dumper( *log, func.GetParamTraits( j ) );
			dumper << curPopulation[i][j];
		}
		*log << "\n";
	}
	*log << "<<<<<<<<DiffEvolution<<<<<<<<--------\n";
}

}
