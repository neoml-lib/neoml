# Miscellaneous Algorithms

<!-- TOC -->

- [Miscellaneous Algorithms](#miscellaneous-algorithms)
	- [Differential evolution](#differential-evolution)
		- [Initial population](#initial-population)
		- [Crossover and mutation](#crossover-and-mutation)
		- [Selection](#selection)
		- [Stop criteria](#stop-criteria)
		- [The function to be optimized](#the-function-to-be-optimized)
		- [Sample code](#sample-code)
	- [Hypotheses generation](#hypotheses-generation)
		- [CGraphGenerator](#cgraphgenerator)
		- [CMatchingGenerator](#cmatchinggenerator)
		- [CSimpleGenerator](#csimplegenerator)

<!-- TOC -->

In this section you will find the description of some **NeoML** machine learning algorithms that do not belong to the other sections.

## Differential evolution

This method finds the global minimum (or maximum) of a non-differentiable, non-linear, multimodal function of many variables F(x1, x2, ..., xn).

It uses mutation, crossover and selection to transform the current population (that is, the function parameters) into the next generation so that the function values on the new population "improve." The process is repeated until the stop criteria are met.

### Initial population

The initial population (x11, x12, ..., x1n) (x21, x22, ..., x2n) ... (xk1, xk2, ..., xkn) is set randomly.

### Crossover and mutation

Let the current generation be

```
(x11, x12, ..., x1n) (x21, x22, ..., x2n) ... (xk1, xk2, ..., xkn)
```

and the next generation be

```
(y11, y12, ..., y1n) (y21, y22, ..., y2n) ... (yk1, yk2, ..., ykn).
```

Then crossover and mutation is performed according to the formula:

```
yij = xij | ( cij + fl * (aij - bij) ),
```

where `a`, `b`, `c` are random vectors from the current population.

### Selection

The current element will be exchanged for the next-generation element with the same number if the optimized function gives a better value on it:

```
yi = F(xi) < F(yi) ? xi : yi
```

### Stop criteria

The algorithm stops if any of the conditions is met:

- the number of iterations reaches the limit
- the function minimum has not changed for a long time
- the set has degenerated

### The function to be optimized

The interface for the function:

```c++
class NEOML_API IFunctionEvaluation {
public:
	// The number of dimensions
	virtual int NumberOfDimensions() const = 0;

	// The parameter "types"
	virtual const IParamTraits& GetParamTraits( int index ) const = 0;
	// The result types
	virtual const IParamTraits& GetResultTraits() const  = 0;

	// Retrieves the maximum and minimum values for the parameters
	virtual CFunctionParam GetMinConstraint( int index ) const = 0;
	virtual CFunctionParam GetMaxConstraint( int index ) const = 0;

	// One of the Evaluate functions should be overloaded in your implementation of this interface
	// Evaluate the function on several parameter sets 
	// (the default implementation calls the one-parameter variant several times)
	virtual void Evaluate( const CArray<CFunctionParamVector>& params, CArray<CFunctionParam>& results );
	// Evaluate the function on one parameter set
	// (the default implementation calls the several-parameters variant) virtual CFunctionParam Evaluate( const CFunctionParamVector& param );
};
```

### Sample code

Here is a sample that runs the algorithm:

```c++
double fluctuation = 0.5; // fluctuation coefficient
double crossProbability = 0.5; // mutation probability
const int populationSize = 100; // population size

CDifferentialEvolution evolution( func, fluctuation, crossProbability, populationSize );
evolution.SetMaxGenerationCount( 200 );
evolution.SetMaxNonGrowingBestValue( 10 );

evolution.RunOptimization();

evolution.GetOptimalVector();
```

## Hypotheses generation

The algorithms described below can generate hypothesis sets that can be used to iterate through various options.

### CGraphGenerator

This algorithm generates paths in a directed acyclic graph. In this graph, each of the arcs should have a quality estimate, and in each node the arcs should be sorted in order of decreasing quality of the path to the end of the graph.

The generator creates a set of paths from the start to the end of the graph, sorted in order of decreasing quality.

#### Sample

This sample creates an instance of the generator and generates the first path.

```c++
CGraphGenerator<CGraph, CArc, int> generator( &graph );
generator.SetMaxStepsQueueSize( 1024 );

CArray<const CArc*> path;
generator.GetNextPath( path );
```

### CMatchingGenerator

The algorithm generates matchings in a bipartite graph. The graph is defined by a penalty matrix for the vertices matchings. Additional penalties apply for skipping the vertices of one part.

The generated matchings are sorted in order of decreasing quality.

#### Sample

This sample creates an instance of the generator and generates the best matching.

```c++
CMatchingGenerator<CEdge, double> generator( leftSize, rightSize );

initializeMatrix( generator.PairMatrix() );
initializeLeftMissedElements( generator.MissedLeftElementPairs() );
initializeRightMissedElements( generator.MissedRightElementPairs() );

generator.Build();

CArray<CEdge> nextMatching;
generator.GetNextMatching( nextMatching );
```

### CSimpleGenerator

This algorithm generates fixed-length sequences. The input data contains a set of arrays with alternative elements, each sorted in order of decreasing quality.

The algorithm generates a set of element variants sorted in order of decreasing total quality.

#### Sample

This sample generates 5-element sequences of integers.

```c++
const int NumberOfElement = 5;
const int NumberOfVariant = 3;

class CIntElement {
public:
	typedef int Quality;

	CIntElement() : value( 0 ) {}
	explicit CIntElement( int _value ) : value( _value ) {}

	int VariantQuality() const { return value; }

private:
	int value;
};

class CIntSimpleGenerator : public CSimpleGenerator<CIntElement> {
public:
	CIntSimpleGenerator() : 
		CSimpleGenerator<CIntElement>( 0, -10000 )
	{
		Variants.SetSize( NumberOfElement );
		for( int i = NumberOfVariant; i > 0; i-- ) {
			for( int j = 0; j < NumberOfElement; j++ ) {
				Variants[j].Add( CIntElement( i ) );
			}
		}
	}
};

CIntSimpleGenerator generator;

CArray<CIntElement> next;
generator.GetNextSet( next );
generator.GetNextSet( next );

```
