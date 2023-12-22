/* Copyright © 2023-2024 ABBYY

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

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <NeoML/Dnn/Layers/FavorAttentionPerformerLayer.h>
#include <NeoMathEngine/NeoMathEngine.h>

namespace NeoML {

// Very big value for normalization's denominator
static constexpr float bigConstant = 1e8f;
// Small positive constant for numerical stability, used to bound away from zero kernel values
static constexpr float numericalStabilizer = 0.001f;
// M_PI value from math.h
static constexpr double piValue = 3.14159265358979323846;

//---------------------------------------------------------------------------------------------------------------------

// Favor Attention Performer descriptor
struct CFavorAttentionDesc final {
public:
	CFavorAttentionDesc( IMathEngine& mathEngine,
		const CBlobDesc& qDesc, const CBlobDesc& kDesc, const CBlobDesc& vDesc, const CBlobDesc& outputDesc,
		int randomFeaturesCount, CFavorAttentionPerformerLayer::TAKernel activation, bool causal );

	// Computes FAVOR normalized attention
	void FavorAttention( const CConstFloatHandle& query, const CConstFloatHandle& key, const CConstFloatHandle& value,
		const CFloatHandle& output );
	// Computes gradient of FAVOR normalized attention
	void FavorAttentionBackward(
		const CConstFloatHandle& query, const CConstFloatHandle& key, const CConstFloatHandle& value,
		const CFloatHandle& queryDiff, const CFloatHandle& keyDiff, const CFloatHandle& valueDiff,
		const CConstFloatHandle& output );

private:
	IMathEngine& mathEngine;

	const int B;
	const int LQ; // seqQ
	const int L;  // seqTo
	const int H;
	const int M;
	const int D;

	const int dim; // number of rows and columns of the resulting 2D-tensor
	const int randomFeaturesCount; // Number of random features to be used (relevant only if projection_matrix set)
	const CFavorAttentionPerformerLayer::TAKernel activation; // Transformation produces kernel features for attention
	const bool causal; // whether attention is auto-regressive or not
	const bool projectionMatrixType; // Either random projection matrix will be applied (for SoftMax should be true)

	CFloatHandleVar projectionTransposedMatrix;
	CFloatHandleVar ratio;
	CFloatHandleVar reluUpperThreshold;
	CFloatHandleVar dataNormalizer;
	CFloatHandleVar numericalStabilizer;
	CFloatHandleVar constHalf;

	// Computes random features for the ReLU-kernel from
	// Args:
	//     input: input data tensor of the shape [B,H,L,D], where:
	//          B - batch dimension, H - heads, L - attention dimensions, D - features.
	//	   output: corresponding kernel feature map.
	//     projection: random Gaussian matrix of shape [M,D], where M stands
	//			for the number of random features and each DxD sub-block has pairwise orthogonal rows.
	// Returns corresponding kernel feature map.
	void reluKernel( const CConstFloatHandle& in, const CFloatHandle& out, const CConstFloatHandle* proj, bool isQuery );
	void reluKernelBackward( const CConstFloatHandle& input, CConstFloatHandle& outputDiff, const CFloatHandle& inputDiff,
		const CConstFloatHandle* projection, bool isQuery );
	// Computes random features for the softmax kernel using FAVOR+ mechanism from
	// Args:
	//     input: input data tensor of the shape [B,H,L,D], where:
	//          B - batch dimension, H - heads, L - attention dimensions, D - features.
	//	   output: corresponding kernel feature map.
	//     projection: random Gaussian matrix of shape [M,D], where M stands
	//			for the number of random features and each DxD sub-block has pairwise orthogonal rows.
	//     query: indicates whether input data is a query or key tensor.
	void softmaxKernel( const CConstFloatHandle& in, const CFloatHandle& out, const CConstFloatHandle* proj, bool isQuery );
	void softmaxKernelBackward( const CConstFloatHandle& input, CConstFloatHandle& outputDiff, const CFloatHandle& inputDiff,
		const CConstFloatHandle* projection, bool isQuery );

	// Constructs random Q matrix of QR-factorization random a 2D-tensor, initilizaed by normal distribution.
	// Args :
	//     Q -- returned result matrix
	//     seed -- randomization initialization
	//     mean and sigma -- parameters of generation float random values
	void createFromUnstructuredBlock( const CFloatHandle& Q, int seed, float mean = 0.f, float sigma = 1.f );
	// Constructs a 2D-tensor which is a product of the form G_1 * ... * G_k, where G_i is a Givens random rotation.
	// The resulting tensor mimics a matrix taken uniformly at random form the orthogonal group.
	// Args :
	//     Q -- returned result matrix
	//     seed -- randomization initialization
	//     min and max -- parameters of generation float random values
	void createProductsOfGivensRotations( const CFloatHandle& Q, int seed, float min = FLT_MIN, float max = FLT_MAX );
	//
	void createMultiplier( const CFloatHandle& multiplier, bool scaling, int seed, float mean = 0.f, float sigma = 1.f );
	// Constructs a matrix of random orthogonal projections.
	// Each projection vector has direction chosen uniformly at randomand either deterministic length \sqrt{ d }
	//		or length taken from the \chi( d ) distribution (in the latter case marginal distributions of the projections are
	//      d - dimensional Gaussian vectors with associated identity covariance matrix ).
	// Args:
	//     m: number of random projections.
	//     d: dimensionality of each random projection.
	//     seed: random seed used to construct projections.
	//     scaling: True if all the random projections need to be renormalized to have
	//			length \sqrt{ d }, False if the lengths of random projections should follow \chi( d ) distribution.
	//     structMode: if True then products of Givens rotations will be used to construct random orthogonal matrix.
	//			This bypasses Gram-Schmidt orthogonalization.
	// Returns the matrix of random projections of the shape[m, d].
	void createProjectionMatrix( const CFloatHandle& projectionMatrixTransposed, int seed = 0,
		bool scaling = CFavorAttentionPerformerLayer::Scaling,
		CFavorAttentionPerformerLayer::TRandomMaxrixStructMode structMode = CFavorAttentionPerformerLayer::StructMode );

	// Computes not-normalized FAVOR noncausal attention AV.
	// Args: query and key tensors of a shape [B,H,M,L] and value tensor of a shape [B,H,D,L].
	// Returns Not - normalized FAVOR noncausal attention AV of a shape [B,H,M,D].
	void nonCausalNumerator( const CConstFloatHandle& qs, const CConstFloatHandle& ks, const CConstFloatHandle& vs,
		const CFloatHandle& result, const CFloatHandle& temp );
	void nonCausalNumeratorBackward( const CConstFloatHandle& qs, const CConstFloatHandle& ks, const CConstFloatHandle& vs,
		const CFloatHandle& result, const CFloatHandle& temp );

	// Computes FAVOR normalizer in noncausal attention.
	// Args: query and key tensors of a shape [B,H,M,L].
	// Returns FAVOR normalizer in noncausal attention of a shape [B,H,L].
	void nonCausalDenominator( const CConstFloatHandle& qs, const CConstFloatHandle& ks,
		const CFloatHandle& result, const CFloatHandle& temp );
	void nonCausalDenominatorBackward( const CConstFloatHandle& qs, const CConstFloatHandle& ks,
		const CFloatHandle& result, const CFloatHandle& temp );

	//Computes not-normalized FAVOR causal attention A_{masked}V.
	// Args: query and key tensors of a shape [B,H,M,L] and value tensor of a shape [B,H,D,L].
	// Returns Not - normalized FAVOR causal attention A_{masked}V of a shape [B,H,M,D].
	void causalNumerator( const CConstFloatHandle& qs, const CConstFloatHandle& ks, const CConstFloatHandle& vs,
		const CFloatHandle& result, const CFloatHandle& temp );
	void causalNumeratorBackward( const CConstFloatHandle& qs, const CConstFloatHandle& ks, const CConstFloatHandle& vs,
		const CFloatHandle& result, const CFloatHandle& temp );

	//Computes FAVOR normalizer in causal attention
	// Args: query and key tensors of a shape [B,H,M,L].
	// Returns FAVOR normalizer in causal attention of a shape [B,H,L].
	void causalDenominator( const CConstFloatHandle& qs, const CConstFloatHandle& ks,
		const CFloatHandle& result, const CFloatHandle& temp );
	void causalDenominatorBackward( const CConstFloatHandle& qs, const CConstFloatHandle& ks,
		const CFloatHandle& result, const CFloatHandle& temp );
};

//---------------------------------------------------------------------------------------------------------------------

CFavorAttentionDesc::CFavorAttentionDesc( IMathEngine& mathEngine,
		const CBlobDesc& qDesc, const CBlobDesc& kDesc, const CBlobDesc& vDesc, const CBlobDesc& outputDesc,
		int randomFeaturesCount, CFavorAttentionPerformerLayer::TAKernel activation, bool causal ) :
	mathEngine( mathEngine ),
	B( qDesc.BatchWidth() ),
	LQ( qDesc.Width() ), // seqQ
	L( vDesc.Width() ),  // seqTo
	H( qDesc.ListSize() ), //heads_count
	M( qDesc.Channels() ), // data_per_head
	D( vDesc.Channels() ), // data_per_head
	dim( M ),
	randomFeaturesCount( randomFeaturesCount ),
	activation( activation ),
	causal( causal ),
	projectionMatrixType( randomFeaturesCount > 0 ),
	projectionTransposedMatrix( mathEngine, projectionMatrixType ? ( randomFeaturesCount * dim ) : 0 ),
	ratio( mathEngine ),
	reluUpperThreshold( mathEngine ),
	dataNormalizer( mathEngine ),
	numericalStabilizer( mathEngine ),
	constHalf( mathEngine )
{
	NeoAssert( B == kDesc.BatchWidth() );
	NeoAssert( B == vDesc.BatchWidth() );
	NeoAssert( H == kDesc.ListSize() );
	NeoAssert( H == vDesc.ListSize() );
	NeoAssert( L == kDesc.Width() );
	NeoAssert( M == kDesc.Channels() );

	NeoAssert( outputDesc.BatchWidth() == B );
	NeoAssert( outputDesc.ListSize() == H );
	NeoAssert( outputDesc.Width() == LQ );
	NeoAssert( outputDesc.Channels() == D );

	ratio.SetValue( static_cast<float>( 1. / std::sqrt( randomFeaturesCount ) ) );
	reluUpperThreshold.SetValue( 0.f );
	dataNormalizer.SetValue( static_cast<float>( 1. / std::sqrt( std::sqrt( dim ) ) ) );
	numericalStabilizer.SetValue( NeoML::numericalStabilizer );
	constHalf.SetValue( 0.5f );
}

void CFavorAttentionDesc::createFromUnstructuredBlock( const CFloatHandle& Q, int seed, float mean, float sigma )
{
	CRandom random( seed );
	const int unstructuredBlockSize = dim * dim;
	const CFloatHandle& unstructuredBlock = Q;
	for( int i = 0; i < unstructuredBlockSize; ++i ) {
		unstructuredBlock.SetValueAt( i, static_cast<float>( random.Normal( mean, sigma ) ) );
	}
	CFloatHandleStackVar qTransposed( mathEngine, static_cast<size_t>( unstructuredBlockSize ) );
	CFloatHandle qTransposedHandle = qTransposed.GetHandle();
	mathEngine.QRFactorization( dim, dim, unstructuredBlock, &qTransposedHandle,
		/*R*/nullptr, /*inplace*/false, /*returnQ*/true, /*returnR*/false );
	mathEngine.TransposeMatrix( /*batchSize*/1, qTransposedHandle,
		/*height*/dim, /*mid*/1, /*width*/dim, /*channels*/1, Q, unstructuredBlockSize );
}

void CFavorAttentionDesc::createProductsOfGivensRotations( const CFloatHandle& Q, int seed, float min, float max )
{
	CRandom random( seed );
	auto getQ = [&]( int i, int j ) -> float { return Q.GetValueAt( i * dim + j ); };
	auto setQ = [&]( int i, int j, float v ) { Q.SetValueAt( i * dim + j, v ); };

	const int numGivensRotations = static_cast<int>( dim * std::ceil( std::log( dim ) ) );
	for( int i = 0; i < numGivensRotations; ++i ) {
		const float randomAngle = static_cast<float>( piValue * random.Uniform( min, max ) );
		const float sinA = std::sin( randomAngle );
		const float cosA = std::cos( randomAngle );

		const int randomIndexMin = random.UniformInt( 0, dim - 1 );
		const int randomIndexMax = random.UniformInt( randomIndexMin + 1, dim );
		for( int j = 0; j < dim; ++j ) {
			const float tmpMin = cosA * getQ( randomIndexMin, j ) + sinA * getQ( randomIndexMax, j );
			const float tmpMax = -sinA * getQ( randomIndexMin, j ) + cosA * getQ( randomIndexMax, j );
			setQ( randomIndexMin, j, tmpMin );
			setQ( randomIndexMax, j, tmpMax );
		}
	}
}

void CFavorAttentionDesc::createMultiplier( const CFloatHandle& multiplier, bool scaling, int seed, float mean, float sigma )
{
	if( scaling == true ) {
		mathEngine.VectorFill( multiplier, static_cast<float>( std::sqrt( dim ) ), randomFeaturesCount );
	} else if( scaling == false ) {
		CFloatHandleStackVar tempUnstructuredBlock( mathEngine, static_cast<size_t>( dim ) );
		CFloatHandle unstructuredBlock = tempUnstructuredBlock.GetHandle();

		CArray<float> values;
		values.SetSize( dim );

		CRandom random( seed );
		for( int feature = 0; feature < randomFeaturesCount; ++feature ) {
			for( int i = 0; i < dim; ++i ) {
				values[i] = static_cast<float>( random.Normal( mean, sigma ) );
			}
			mathEngine.DataExchangeRaw( unstructuredBlock, values.GetPtr(), dim );
			mathEngine.VectorEltwiseMultiply( unstructuredBlock, unstructuredBlock, unstructuredBlock, dim );
			mathEngine.VectorSum( unstructuredBlock, dim, multiplier + feature );
		}
		mathEngine.VectorSqrt( multiplier, multiplier, randomFeaturesCount );
	}
}

void CFavorAttentionDesc::createProjectionMatrix( const CFloatHandle& projectionMatrixTransposed, int seed,
	bool scaling, CFavorAttentionPerformerLayer::TRandomMaxrixStructMode structMode )
{
	const int m = randomFeaturesCount;
	const size_t qSize = static_cast<size_t>( dim ) * dim;
	const int finalSize = m * dim;
	CFloatHandleStackVar finalMatrix( mathEngine, static_cast<size_t>( finalSize ) );

	const int numFullBlocks = m / dim;
	const int remainingRows = m - numFullBlocks * dim;
	int current_seed = seed;
	CFloatHandle Q = finalMatrix.GetHandle();
	for( int i = 0; i < numFullBlocks; ++i ) {
		if( structMode == CFavorAttentionPerformerLayer::TRandomMaxrixStructMode::QMatrix ) {
			createProductsOfGivensRotations( Q, seed );
		} else {
			createFromUnstructuredBlock( Q, current_seed++ );
		}
		Q += qSize;
	}

	if( remainingRows > 0 ) {
		CFloatHandleStackVar tempQVar( mathEngine, qSize );
		CFloatHandle tempQ = tempQVar.GetHandle();
		if( structMode == CFavorAttentionPerformerLayer::TRandomMaxrixStructMode::QMatrix ) {
			createProductsOfGivensRotations( tempQ, seed );
		} else {
			createFromUnstructuredBlock( tempQ, current_seed++ );
		}
		mathEngine.VectorCopy( Q, tempQ, remainingRows * dim );
	}

	CFloatHandleStackVar projectionMatrix( mathEngine, static_cast<size_t>( finalSize ) );
	{
		CFloatHandleStackVar mulVal( mathEngine, static_cast<size_t>( m ) );
		CFloatHandle multiplier = mulVal.GetHandle();
		createMultiplier( multiplier, scaling, current_seed );

		mathEngine.MultiplyDiagMatrixByMatrix( multiplier, m, finalMatrix, dim, projectionMatrix, finalSize );
	}
	mathEngine.TransposeMatrix( /*batchSize*/1, projectionMatrix,
		/*height*/m, /*mid*/1, /*width*/dim, /*channels*/1, projectionMatrixTransposed, finalSize );
}

void CFavorAttentionDesc::nonCausalNumerator( const CConstFloatHandle& qs, const CConstFloatHandle& ks,
	const CConstFloatHandle& vs, const CFloatHandle& result, const CFloatHandle& kvs )
{
	mathEngine.MultiplyTransposedMatrixByMatrix( B * H, //bhlm,bhld->bhmd
		ks, L, M,
		vs, D,
		kvs, B * H * M * D );
	mathEngine.MultiplyMatrixByMatrix( B * H, //bhlm,bhmd->bhld
		qs, LQ, M,
		kvs, D,
		result, B * H * LQ * D );
}

void CFavorAttentionDesc::nonCausalNumeratorBackward( const CConstFloatHandle& /*qs*/, const CConstFloatHandle& /*ks*/,
	const CConstFloatHandle& /*vs*/, const CFloatHandle& /*result*/, const CFloatHandle& /*temp*/ )
{
	NeoAssert( false );
}

void CFavorAttentionDesc::nonCausalDenominator( const CConstFloatHandle& qs, const CConstFloatHandle& ks,
	const CFloatHandle& result, const CFloatHandle& ksum )
{
	//all_ones = tf.ones( [ks.shape[0]] ); //[L,B,H,M]
	CFloatHandleStackVar allOnes( mathEngine, L );
	mathEngine.VectorFill( allOnes, 1, L );

	CFloatHandle ksumPtr = ksum;
	CConstFloatHandle ksPtr = ks;
	for( int b = 0; b < B * H; ++b ) {
		mathEngine.MultiplyTransposedMatrixByMatrix( /*batchSize*/1, //bhlm,l->bhm
			ksPtr, L, M,
			allOnes, 1,
			ksumPtr, B * H * M );
		ksPtr += L * M;
		ksumPtr += M;
	}
	mathEngine.MultiplyMatrixByMatrix( B * H, //bhlm,bhm->bhl
		qs, LQ, M,
		ksum, 1,
		result, B * H * LQ );
}

void CFavorAttentionDesc::nonCausalDenominatorBackward( const CConstFloatHandle& /*qs*/, const CConstFloatHandle& /*ks*/,
	const CFloatHandle& /*result*/, const CFloatHandle& /*temp*/ )
{
	NeoAssert( false );
}

void CFavorAttentionDesc::causalNumerator( const CConstFloatHandle& /*qs*/, const CConstFloatHandle& /*ks*/,
	const CConstFloatHandle& /*vs*/, const CFloatHandle& /*result*/, const CFloatHandle& )
{
	NeoAssert( false ); // TODO
}

void CFavorAttentionDesc::causalNumeratorBackward( const CConstFloatHandle& /*qs*/, const CConstFloatHandle& /*ks*/,
	const CConstFloatHandle& /*vs*/, const CFloatHandle& /*result*/, const CFloatHandle& /*temp*/ )
{
	NeoAssert( false );
}

void CFavorAttentionDesc::causalDenominator( const CConstFloatHandle& /*qs*/, const CConstFloatHandle& /*ks*/,
	const CFloatHandle& /*result*/, const CFloatHandle& )
{
	NeoAssert( false );
}

void CFavorAttentionDesc::causalDenominatorBackward( const CConstFloatHandle& /*qs*/, const CConstFloatHandle& /*ks*/,
	const CFloatHandle& /*result*/, const CFloatHandle& /*temp*/ )
{
	NeoAssert( false );
}

void CFavorAttentionDesc::FavorAttention(
	const CConstFloatHandle& query, const CConstFloatHandle& key, const CConstFloatHandle& value,
	const CFloatHandle& output )
{
	CConstFloatHandle projectionTransposedConstHandle = projectionTransposedMatrix.GetHandle();
	CConstFloatHandle* projectionTransposed = nullptr;
	if( projectionMatrixType == true ) {
		CFloatHandleStackVar reduce( mathEngine );
		mathEngine.VectorSum( query, B * H * LQ * M, reduce );
		const int seed = static_cast<int>( std::ceil( std::abs( reduce.GetValue() * NeoML::bigConstant ) ) );
		CFloatHandle projectionTransposedHandle = projectionTransposedMatrix.GetHandle();
		createProjectionMatrix( projectionTransposedHandle, seed );
		projectionTransposed = &projectionTransposedConstHandle;
	}

	CFloatHandleStackVar tempQueryPrime( mathEngine, B * H * LQ * M );
	CFloatHandleStackVar tempKeyPrime( mathEngine, B * H * L * M );

	CFloatHandle queryPrime = tempQueryPrime;
	CFloatHandle keyPrime = tempKeyPrime;

	if( activation == CFavorAttentionPerformerLayer::TAKernel::SoftMax ) {
		softmaxKernel( query, queryPrime, projectionTransposed, true );  //[B,H,L,M]
		softmaxKernel( key, keyPrime, projectionTransposed, false );  //[B,H,L,M]
	} else {
		NeoAssert( activation == CFavorAttentionPerformerLayer::TAKernel::ReLU );
		reluKernel( query, queryPrime, projectionTransposed, true );  //[B,H,L,M]
		reluKernel( key, keyPrime, projectionTransposed, false );  //[B,H,L,M]
	}

	CFloatHandleStackVar temp( mathEngine, B * H * M * D );
	CFloatHandle attentionNorm = tempKeyPrime;

	if( causal ) {
		causalNumerator( queryPrime, keyPrime, value, output, temp ); //[B,H,LQ,D]
		causalDenominator( queryPrime, keyPrime, attentionNorm, temp ); //[B,H,LQ]
	} else {
		nonCausalNumerator( queryPrime, keyPrime, value, output, temp ); //[B,H,LQ,D]
		nonCausalDenominator( queryPrime, keyPrime, attentionNorm, temp ); //[B,H,LQ]
	}
	mathEngine.MatrixColumnsEltwiseDivide( output, /*height*/( B * H * LQ ), /*width*/D, attentionNorm, output );
}

void CFavorAttentionDesc::FavorAttentionBackward(
	const CConstFloatHandle& /*query*/, const CConstFloatHandle& /*key*/, const CConstFloatHandle& /*value*/,
	const CFloatHandle& /*queryDiff*/, const CFloatHandle& /*keyDiff*/, const CFloatHandle& /*valueDiff*/,
	const CConstFloatHandle& /*output*/ )
{
	NeoAssert( false );
}

void CFavorAttentionDesc::reluKernel( const CConstFloatHandle& input, const CFloatHandle& output,
	const CConstFloatHandle* projectionTransposed, bool isQuery )
{
	const int LZ = isQuery ? LQ : L;
	const int size = B * H * LZ * M;
	if( projectionTransposed == nullptr ) {
		mathEngine.VectorReLU( input, output, size, reluUpperThreshold );
		mathEngine.VectorAddValue( output, output, size, numericalStabilizer );
	} else {
		CConstFloatHandle inputPtr = input;
		CFloatHandle outputPtr = output;
		for( int i = 0; i < B * H; ++i ) {
			mathEngine.MultiplyMatrixByTransposedMatrix( /*batchSize*/1, //bhlm,dm->bhld
				inputPtr, LZ, M,
				*projectionTransposed, dim,
				outputPtr, size );
			inputPtr += LZ * M;
			outputPtr += LZ * dim;
		}
		mathEngine.VectorMultiply( output, output, size, ratio );
		mathEngine.VectorReLU( output, output, size, reluUpperThreshold );
		mathEngine.VectorAddValue( output, output, size, numericalStabilizer );
	}
}

void CFavorAttentionDesc::reluKernelBackward( const CConstFloatHandle& input, CConstFloatHandle& outputDiff,
	const CFloatHandle& inputDiff, const CConstFloatHandle* projectionTransposed, bool isQuery )
{
	const int LZ = isQuery ? LQ : L;
	const int size = B * H * LZ * M;
	if( projectionTransposed == nullptr ) {
		mathEngine.VectorReLUDiffOp( /*outputBlob*/input, outputDiff, inputDiff, size, reluUpperThreshold );
	} else {
		NeoAssert( false );
	}
}

void CFavorAttentionDesc::softmaxKernel( const CConstFloatHandle& input, const CFloatHandle& output,
	const CConstFloatHandle* projectionTransposed, bool isQuery )
{
	const int LZ = isQuery ? LQ : L;
	const int size = B * H * LZ * M;
	ASSERT_EXPR( projectionTransposed != nullptr );

	CFloatHandleStackVar tempInput( mathEngine, size );
	mathEngine.VectorMultiply( input, tempInput, size, dataNormalizer );

	CConstFloatHandle inputPtr = tempInput;
	CFloatHandle outputPtr = output;
	for( int i = 0; i < B * H; ++i ) {
		mathEngine.MultiplyMatrixByTransposedMatrix( /*batchSize*/1, //bhlm,dm->bhld
			inputPtr, LZ, M,
			*projectionTransposed, dim,
			outputPtr, size );
		inputPtr += LZ * M;
		outputPtr += LZ * dim;
	}
	mathEngine.VectorEltwiseMultiply( tempInput, tempInput, tempInput, size );

	const int reduceSize = size / M;
	CFloatHandleStackVar tempDiag( mathEngine, reduceSize );
	mathEngine.VectorSumAlongDimension( tempInput, /*before*/reduceSize, /*size*/M, /*after*/1, tempDiag );
	mathEngine.VectorMultiply( tempDiag, tempDiag, reduceSize, constHalf );

	CFloatHandleStackVar tempReduceMax( mathEngine, reduceSize );
	mathEngine.FindMaxValueInRows( output, reduceSize, M, tempReduceMax, reduceSize );
	if( !isQuery ) {
		mathEngine.FindMaxValueInRows( tempReduceMax, B * H, L, tempReduceMax, reduceSize );
		mathEngine.SubVectorFromMatrixColumns( output, output, B * H, L * M, tempReduceMax );
	} else {
		mathEngine.SubVectorFromMatrixColumns( output, output, reduceSize, M, tempReduceMax );
	}
	mathEngine.SubVectorFromMatrixColumns( output, output, reduceSize, M, tempDiag );
	mathEngine.VectorExp( output, output, size );

	mathEngine.VectorAddValue( output, output, size, numericalStabilizer );
	mathEngine.VectorMultiply( output, output, size, ratio );
}

void CFavorAttentionDesc::softmaxKernelBackward( const CConstFloatHandle& /*input*/, CConstFloatHandle& /*outputDiff*/,
	const CFloatHandle& /*inputDiff*/, const CConstFloatHandle* /*projectionTransposed*/, bool /*isQuery*/ )
{
	NeoAssert( false );
}

//---------------------------------------------------------------------------------------------------------------------

CFavorAttentionPerformerLayer::CFavorAttentionPerformerLayer( IMathEngine& mathEngine, const char* name ) :
	CBaseLayer( mathEngine, ( name == nullptr ) ? "CDnnFavorAttentionPerformerLayer" : name, /*isLearnable*/false )
{}

CFavorAttentionPerformerLayer::~CFavorAttentionPerformerLayer()
{
	destroyFavorAttentionDesc();
}

void CFavorAttentionPerformerLayer::destroyFavorAttentionDesc()
{
	if( desc != nullptr ) {
		delete desc;
		desc = nullptr;
	}
}

static const int FavorAttentionPerformerLayerVersion = 0;

void CFavorAttentionPerformerLayer::Serialize( CArchive& archive )
{
	( void ) archive.SerializeVersion( FavorAttentionPerformerLayerVersion );
	CBaseLayer::Serialize( archive );

	archive.Serialize( randomFeaturesCount );
	archive.SerializeEnum( activation );
	archive.Serialize( causal );

	if( archive.IsLoading() ) {
		destroyFavorAttentionDesc();
	}
}

void CFavorAttentionPerformerLayer::SetRandomFeaturesCount( int _randomFeaturesCount )
{
	NeoAssert( _randomFeaturesCount >= 0 );
	if( randomFeaturesCount != _randomFeaturesCount ) {
		randomFeaturesCount = _randomFeaturesCount;
		destroyFavorAttentionDesc();
	}
}

void CFavorAttentionPerformerLayer::SetActivationKernel( int _activation )
{
	TAKernel newActivation = static_cast<TAKernel>( _activation );
	NeoAssert( newActivation == TAKernel::SoftMax || newActivation == TAKernel::ReLU );

	if( activation != newActivation ) {
		activation = newActivation;
		destroyFavorAttentionDesc();
	}
}

void CFavorAttentionPerformerLayer::SetCausal( bool _causal )
{
	if( causal != _causal ) {
		causal = _causal;
		destroyFavorAttentionDesc();
	}
}

void CFavorAttentionPerformerLayer::Reshape()
{
	CheckInputs();
	CheckLayerArchitecture( GetInputCount() == 3, "Favor Attention layer inputs count should be 3" );
	CheckLayerArchitecture( GetOutputCount() == 1, "Favor Attention layer outputs count should be 1" );

	// For each layer element there is a channel in the output blob
	outputDescs[0] = inputDescs[TI_Q];
	outputDescs[0].SetDimSize( BD_Channels, inputDescs[TI_V].Channels() );

	destroyFavorAttentionDesc();
}

void CFavorAttentionPerformerLayer::RunOnce()
{
	if( desc == nullptr ) {
		desc = new CFavorAttentionDesc( MathEngine(),
			inputBlobs[TI_Q]->GetDesc(), inputBlobs[TI_K]->GetDesc(), inputBlobs[TI_V]->GetDesc(),
			outputBlobs[0]->GetDesc(), randomFeaturesCount, activation, causal );
	}

	CFloatHandle query = inputBlobs[TI_Q]->GetData(); // [B, n_head, seq_Q, d_k]
	CFloatHandle key = inputBlobs[TI_K]->GetData();   // [B, n_head, seq_to, d_k]
	CFloatHandle value = inputBlobs[TI_V]->GetData(); // [B, n_head, seq_to, d_k]
	CFloatHandle output = outputBlobs[0]->GetData();  // [B, n_head, seq_Q, d_k]

	// Linearly project the query, key and value using different learned projections.
	// Splitting heads is automatically done during the linear projections -->
	//    [batchSize, sequenceLength, headCount, sizePerHead] --> (names = [B,L,H,M])
	//    (Tensor's [BatchWidth, ListSize, Width, Channels] inited)
	desc->FavorAttention( query, key, value, output );
}

void CFavorAttentionPerformerLayer::BackwardOnce()
{
	NeoAssert( desc != nullptr );

	CFloatHandle queryDiff = inputDiffBlobs[0]->GetData();
	CFloatHandle keyDiff = inputDiffBlobs[1]->GetData();
	CFloatHandle valueDiff = inputDiffBlobs[2]->GetData();
	CFloatHandle outputDiff = outputDiffBlobs[0]->GetData();

	CConstFloatHandle query = inputBlobs[0]->GetData();
	CConstFloatHandle key = inputDiffBlobs[1]->GetData();
	CConstFloatHandle value = inputDiffBlobs[2]->GetData();

	desc->FavorAttentionBackward( query, key, value, queryDiff, keyDiff, valueDiff, outputDiff );
}

//---------------------------------------------------------------------------------------------------------------------

NEOML_API CLayerWrapper<CFavorAttentionPerformerLayer> FavorAttentionPerformer(
	int randomFeaturesCount, int activation, bool causal )
{
	return CLayerWrapper<CFavorAttentionPerformerLayer>( "FavorAttentionPerformer",
		[=]( CFavorAttentionPerformerLayer* layer ) {
			layer->SetRandomFeaturesCount( randomFeaturesCount );
			layer->SetActivationKernel( activation );
			layer->SetCausal( causal );
		} );
}

} // namespace NeoML
