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
	- [Dimensionality reduction](#dimensionality-reduction)
		- [Principal component analysis](#principal-component-analysis)
	- [Subword encoding for language modelling](#subword-encoding-for-language-modelling)
		- [Byte pair encoding](#byte-pair-encoding)

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
// A function which evalutes a vector of CSvm paramters
// The differential evolution interprets every axis as linear on the [min;max] interval
// That's why we're searching the optimal logoarithm (base 10) of some of the parameters
class CSvmEvaluation : public IFunctionEvaluation {
private:
	// CSvm parameters optimized during diff evolution
	enum TSvmParam {
		SP_KernelType, // Kernel type, enum (which is encoded as int)
		SP_LogErrorWeight, // Logarithm of ErrorWeight parameter
		SP_MaxIterations, // Maximum number of iterations
		SP_Degree, // Degree parameter
		SP_LogGamma, // Logarithm of Gamma parameter
		SP_LogCoeff0, // Logarithm of Coeff0 parameter
		SP_LogTolerance, // Logarithm of Tolerance parameter

		SP_Count // Vector size
	};
public:
	// Accepts the data and the number of folds in cross-validation
	explicit CSvmEvaluation( const IProblem& problem, int cvFolds ) :
		problem( &problem ), cvFolds( cvFolds ) {}

	// IFunctionEvaluation interface 

	// Number of elements in vector of parameters
	int NumberOfDimensions() const override { return static_cast<int>( SP_Count ); }

	// Type of each of the parameter in vector
	const IParamTraits& GetParamTraits( int index ) const override
	{
		switch( static_cast<TSvmParam>( index ) ) {
			case SP_KernelType:
			case SP_MaxIterations:
			case SP_Degree:
				return CIntTraits::GetInstance();
			case SP_LogErrorWeight:
			case SP_LogGamma:
			case SP_LogCoeff0:
			case SP_LogTolerance:
				return CDoubleTraits::GetInstance();
			case SP_Count:
			default:
				NeoAssert( false );
		}
		return CIntTraits::GetInstance();
	}

	// Type of the optimized value 
	const IParamTraits& GetResultTraits() const override { return CDoubleTraits::GetInstance(); }

	// The minimum value of the index'th parameter
	CFunctionParam GetMinConstraint( int index ) const override
	{
		switch( static_cast<TSvmParam>( index ) ) {
			case SP_KernelType:
				return CIntTraits::GetInstance().Box( static_cast<int>( CSvmKernel::KT_Linear ) );
			case SP_LogErrorWeight:
				return CDoubleTraits::GetInstance().Box( -3. );
			case SP_MaxIterations:
				return CIntTraits::GetInstance().Box( 10 );
			case SP_Degree:
				return CIntTraits::GetInstance().Box( 1 );
			case SP_LogGamma:
				return CDoubleTraits::GetInstance().Box( -3. );
			case SP_LogCoeff0:
				return CDoubleTraits::GetInstance().Box( -3. );
			case SP_LogTolerance:
				return CDoubleTraits::GetInstance().Box( -4 );
			default:
				NeoAssert( false );
		}
		return CDoubleTraits::GetInstance().Box( 1 );
	}

	// The maximum value of the index'th parameter
	CFunctionParam GetMaxConstraint( int index ) const override
	{
		switch( static_cast<TSvmParam>( index ) ) {
			case SP_KernelType:
				return CIntTraits::GetInstance().Box( static_cast<int>( CSvmKernel::KT_Sigmoid ) );
			case SP_LogErrorWeight:
				return CDoubleTraits::GetInstance().Box( 3. );
			case SP_MaxIterations:
				return CIntTraits::GetInstance().Box( 1000 );
			case SP_Degree:
				return CIntTraits::GetInstance().Box( 5 );
			case SP_LogGamma:
				return CDoubleTraits::GetInstance().Box( 3 );
			case SP_LogCoeff0:
				return CDoubleTraits::GetInstance().Box( 3 );
			case SP_LogTolerance:
				return CDoubleTraits::GetInstance().Box( -1 );
			default:
				NeoAssert( false );
		}
		return CDoubleTraits::GetInstance().Box( 1 );
	}

	// Evaluates a single parameter vector
	// In this case it returns the average accuracy of the cross-validation of CSvm's with the given param
	// on the data given in constructor
	CFunctionParam Evaluate( const CFunctionParamVector& param ) override
	{
		// Don't forget that some of the parameters are logarithms!
		CSvm::CParams svmParams(
			static_cast<CSvmKernel::TKernelType>( CIntTraits::GetInstance().Unbox( param[SP_KernelType] ) ),
			::pow( 10., CDoubleTraits::GetInstance().Unbox( param[SP_LogErrorWeight] ) ),
			CIntTraits::GetInstance().Unbox( param[SP_MaxIterations] ),
			CIntTraits::GetInstance().Unbox( param[SP_Degree] ),
			::pow( 10., CDoubleTraits::GetInstance().Unbox( param[SP_LogGamma] ) ),
			::pow( 10., CDoubleTraits::GetInstance().Unbox( param[SP_LogCoeff0] ) ),
			::pow( 10., CDoubleTraits::GetInstance().Unbox( param[SP_LogTolerance] ) ),
			true,
			OmpGetMaxThreadCount(),
			MM_OneVsOne
		);

		CSvm svm( svmParams );
		CCrossValidation cv( svm, problem );
		CCrossValidationResult cvResult;
		cv.Execute( cvFolds, AccuracyScore, cvResult, true );

		double total = 0;
		for( int i = 0; i < cvResult.Success.Size(); ++i ) {
			total += cvResult.Success[i];
		}
		// The differential evolution minimizes it's target value
		// But in our task we're trying to maximize the accuracy
		// This is why we're using the negative of it
		return CDoubleTraits::GetInstance().Box( -total / cvResult.Success.Size() );
	}

private:
	CPtr<const IProblem> problem;
	int cvFolds;
};

double fluctuation = 0.5; // fluctuation coefficient
double crossProbability = 0.5; // mutation probability
const int populationSize = 20; // population size

CSvmEvaluation svmEval( *problem, 5 );
CDifferentialEvolution evolution( svmEval, fluctuation, crossProbability, populationSize );
evolution.SetMaxGenerationCount( 100 );
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

## Dimensionality reduction

It may be useful in many tasks to lower the dimensionality of a large multidimensional dataset while still retaining most of the information it contained.

### Principal component analysis

PCA uses singular value decomposition to project the dataset into a lower dimensional space.

```c++
void SingularValueDecomposition( const CFloatMatrixDesc& data, const TSvd& svdSolver,
	CArray<float>& leftVectors, CArray<float>& singularValues, CArray<float>& rightVectors,
	bool returnLeftVectors, bool returnRightVectors, int components )
```
There are two algorithms for full and sparse matrices, `SVD_Full` and `SVD_Sparse` correspondingly.

The number of principal components can be selected in several ways, determined by the `ComponentsType` field of `CParams`:

* `PCAC_None`: the number of components will simply be the smaller of the data matrix width and height
* `PCAC_Int`: the number of components is directly specified in the `Components` field
* `PCAC_Float`: select the number of components so that the explained variance is greater than the float value in `Components` field (it should be in (0, 1) range)

The `Train` method performs SVD, then takes the required number of the singular vectors for principal components, selecting those that correspond to the largest singular values.

The `TrainTransform` method does the same, then transforms the data matrix into the new principal component coordinates.

The `Transform` method only transforms the given data matrix into the selected principal component coordinates.

You can access the singular values, variance, and the principal components via the getter methods.

```c++
class NEOML_API CPca {
public:
    enum TComponents {
		PCAC_None = 0,
		PCAC_Int,
		PCAC_Float,
    	PCAC_Count
	};

	struct CParams {
		TComponents ComponentsType;
		TSvd SvdSolver;
		float Components;

		CParams() :
			ComponentsType( PCAC_None ),
			SvdSolver( SVD_Full ),
			Components( 0 )
		{
		}
	};

	// Chooses the greatest singular values from `Components` and
	// selects the corresponding principal axes as the final components
	void Train( const CFloatMatrixDesc& data );
	// Transforms the data into shape ( samples x components )
	// using the principal components calculated before
	CSparseFloatMatrixDesc Transform( const CFloatMatrixDesc& data );
	// Trains and transforms the data into shape ( samples x components )
	CSparseFloatMatrixDesc TrainTransform( const CFloatMatrixDesc& data );

	// Singular values corresponding to the selected principal axes
	const CArray<float>& GetSingularValues() const { return singularValues; }
	// Variance explained by each of the selected principal axes
	const CArray<float>& GetExplainedVariance() const { return explainedVariance; }
	// Percentage of variance explained by each of the selected principal axis
	const CArray<float>& GetExplainedVarianceRatio() const { return explainedVarianceRatio; }
	// Mean of singular values not corresponding to the selected principal axes
	float GetNoiseVariance() const { return noiseVariance; }
	// Selected number of principal axes
	int GetComponentsNum() const { return components; }
	// Matrix ( components x features ) with rows corresponding to the selected principal axis
	CSparseFloatMatrix GetComponents();

	// Get input params
	CParams GetParams() const { return params; }
	// For serialization
	static CPtr<CPca> Create() { return FINE_DEBUG_NEW CPca(); }
	// Serializes the model
	void Serialize( CArchive& archive ) override;

```

## Subword encoding for language modelling

Subword tokenization (+encoding) is approach which has advantages over character-based and word-based approach in language modeling tasks.

```c++
// Subword encoder interface.
class NEOML_API ISubwordEncoder : virtual public IObject {
public:
	virtual ~ISubwordEncoder() override = default;

	// Encodes a word as a sequence of token ids with corresponding token lengths.
	// TokenId range = [0, ... , Size() - 1].
	// To encode a string with wide characters you have to first encode it as utf-8 and wrap it in CString.
	// In this case 'tokenLengths' will contain lengths of the tokens according to the original string version.
	virtual void Encode( const CString& word, CArray<int>& tokenIds,
		CArray<int>& tokenLengths ) const = 0;
	
	// Decodes sequence of token ids into a sequence of words.
	virtual void Decode( const CArray<int>& tokenIds, CArray<CString>& words ) const = 0;

	// Returns number of tokens.
	virtual int Size() const = 0;
};
```

Some subword encoding algorithms can be accelerated by using a cache for `Encode` calls.
For example, during the training of language model, a large number of `Encode` calls usually occur.

For this reason an additional interface has been added:

```c++
// Subword encoder which supports caching results of 'Encode' calls.
class NEOML_API ISubwordEncoderWithCache : public ISubwordEncoder {
public:
	virtual void Encode( const CString& word, CArray<int>& tokenIds,
		CArray<int>& tokenLengths ) const override final;

	// Sets the cache cleanup period.
	// The cache is used for Encode calls acceleration.
	// The result of the encode call is cached and will be erased if 
	// no call with the same word will occur among next 1-2 X cachePeriod calls.
	// Increase in cachePeriod leads to a in increase in memory consumption.
	// To completely switch the cache off set cachePeriod equal to -1.
	// Value 0 is treated as invalid.
	void SetCachePeriod( int cachePeriod ) const { cache.SetCachePeriod( cachePeriod ); }
```

### Byte pair encoding

Popular subword encoding algorithm.

```c++
class NEOML_API IBytePairEncoder : public ISubwordEncoderWithCache {
public:
	// Returns encoder flags.
	virtual bool UseEndOfWordToken() const = 0;
	virtual bool UseStartOfWordToken() const = 0;

	// Word with vocabulary directly. See below.
	virtual void LoadDictionary( const CWordDictionary& tokens, 
		const CString& endOfWordToken, const CString& startOfWordToken ) = 0;
	virtual void GetDictionary( CWordDictionary& tokens, 
		const CString& endOfWordToken = "</s>", const CString& startOfWordToken = "<s>" ) const = 0;
};
```

Additional methods cover usage of special tokens: End-Of-Word and Start-Of-Word token.
Basically you probably want End-Of-Word set to `True` and Start-Of-Word set to `False`.

To train encoder with `IBytePairEncoder` functionality you need to use `CBytePairEncoderTrainer`:

```c++
// Class that trains byte-pair-encoding.
class NEOML_API CBytePairEncoderTrainer {
public:
	struct CParams {
		// Max size of encoder.
		// The size of the trained encoder cannot exceed this value, but CAN be smaller.
		int MaxSize;
		// Add EoW token to each word.
		bool UseEndOfWordToken;
		// Add SoW token to each word.
		bool UseStartOfWordToken;

		CParams() :
			MaxSize( 50000 ),
			UseEndOfWordToken( true ),
			UseStartOfWordToken( false )
		{}
	};

	CBytePairEncoderTrainer( const CParams& params, const CWordDictionary& dictionary );

	// Trains and returns a fully trained encoder.
	CPtr<IBytePairEncoder> Train();

	// Trains addtional #stepsCount tokens.
	// Returns true if no additional steps can be performed.
	bool TrainSteps( int stepsCount );

	// Returns true if training has been completed.
	bool IsTrainingCompleted() const;

	// Returns encoder consisting of bpe tokens obtained from completed steps.
	CPtr<IBytePairEncoder> GetEncoder() const;

	// Save/load checkpoint
	void Serialize( CArchive& archive );
```

How to use:

1. Create a dictionary of counted words from text corpus using `CWordDictionary` class instance.
2. Create `CBytePairEncoderTrainer` instance with desired `CParams` and the dictionary.
3. Call `Train` method of encoder trainer.
    * Use `TrainSteps` method if you want to perform partial training of encoder. To get partially trained encoder use `GetEncoder` method.


For debug reasons, `IBytePairEncoder` also provides direct methods for loading an externally created dictionary or exporting its own (`LoadDictionary` and `GetDictionary`). An external dictionary must be valid for using with our encoder:
1. Every token except the letters must be a concatenation of two smaller tokens.
2. If used, End-Of-Word can be located only in the end of a token. If used, Start-Of-Word can be located only in the beginning of a token.
3. If used, End-Of-Word and Start-Of-Word must be contained in the dictionary as separate tokens (which is just an implication from rules above).

`IBytePairEncoder` replaces user-defined End-Of-Word and Start-Of-Word marks with special unreadable symbols unlikely to appear in any text. Therefore, the original marks are lost and should be given as arguments of `GetDictionary`.
