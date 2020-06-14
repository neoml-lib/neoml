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

#pragma once

#include <NeoML/NeoMLDefs.h>
#include <NeoML/Random.h>
#include <NeoML/TraditionalML/FunctionEvaluation.h>

namespace NeoML {

// Optimizing function value implementation based on differential evolution
// The purpose of the algorithm is to find the optimal system parameters 
// (represented by a vector of real values X) for which 
// the specified function value f(X) will be the closest to the reference value.
// The Evaluate function to calculate f(X) and compare it with the reference
// is provided by the user.

class NEOML_API CDifferentialEvolution {
public:
	// Parameters:
	//   fluctuation - the coefficient to calculate a mutated value
	//   score - the required approximation of the optimized function
	//   dimension - the number of parameters
	//   cr - the coefficient to calculate elements shuffling
	//   population - the number of elements in a generation
	CDifferentialEvolution( IFunctionEvaluation& func, double fluctuation = 0.5, double cr = 0.5,
		int population = 100 );

	// Sets the first generation values
	void SetFirstGeneration( const CArray<CFunctionParamVector>& generation );
	void SetFirstGeneration( const CArray<CFunctionParamVector>& generation,
		const CArray<CFunctionParam>& generationValues );

	// Sets the maximum number of generations
	void SetMaxGenerationCount( int count ) { maxGenerationCount = count; }
	
	// Sets the maximum number of generations without best value improvement
	void SetMaxNonGrowingBestValue( int count ) { maxNonGrowingBestValue = count; }

	// Builds next generation; returns true if any of the stop conditions was fulfilled
	bool BuildNextGeneration();

	// Runs optimization until one of the stop conditions is fulfilled
	void RunOptimization() { while( !BuildNextGeneration() ); }
	
	// Gets the resulting population
	const CArray<CFunctionParamVector>& GetPopulation() const { return curPopulation; }
	// Gets the function values on the resulting population
	const CArray<CFunctionParam>& GetPopulationFuncValues() const { return funcValues; }
	// Gets the "best vector"
	CFunctionParamVector GetOptimalVector() const;
	
	// Sets a text stream for logging processing
	// By default logging is off (set to null to turn off)
	void SetLog( CTextStream* newLog ) { log = newLog; }

	// Write the current population into the log
	void LogPopulation();

private:
	IFunctionEvaluation& func; // the function to optimize
	CTextStream* log; // the logging stream
	int population; // the number of elements in one generation
	double fluctuation; // the fluctuation coefficient to calculate parameters
	double crossProbability; // the fluctuation coefficient for elements shuffling

	int maxGenerationCount;
	int generationNum; // the number of the current generation

	// The current and the next populations
	CArray<CFunctionParamVector> curPopulation, nextPopulation;
	// The function values for the current population
	CArray<CFunctionParam> funcValues;

	CFunctionParam lastBestValue;	// the best value found
	int lastBestGenerationNum;	// the number of the generation on which the best value was reached
	int maxNonGrowingBestValue;	// the maximum number of generations without best value improvement

	mutable CRandom random;
	
	CFunctionParamVector initPoint() const;

	void initializeAlgo();
	bool checkStop();

	CFunctionParam mutate( const IParamTraits& traits, const CFunctionParam& p,
		const CFunctionParam& a, const CFunctionParam& b, const CFunctionParam& c,
		const CFunctionParam& minVal, const CFunctionParam& maxVal );
};

}
