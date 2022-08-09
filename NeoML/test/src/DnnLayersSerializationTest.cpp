/* Copyright © 2021 ABBYY Production LLC

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

#include <TestFixture.h>

using namespace NeoML;
using namespace NeoMLTest;

static const int TestIntValue = 3;
static const float TestFloatValue = 4.;
static const CString LayerName = "LAYER";
static const int TestSize = 20;

// #define GENERATE_SERIALIZATION_FILES

static const CString getFileName( const CString& name )
{
	return GetTestDataFilePath( "data/LayersSerializationTestData", name + ".arch" );
}

// Checks the coverage of the tests below
GTEST_TEST( LayerSerialization, CheckRegisteredLayers )
{
	CHashTable<CString> ignoredLayers;
	ignoredLayers.Add( "FmlCCnnChannelwiseSoftmaxLayer" ); // It's an alternative name for CSoftmaxLayer
	ignoredLayers.Add( "FmlCnnGlobalMainPoolingLayer" ); // It's an alternative name for CGlobalMeanPoolingLayer

	CArray<const char*> layerClasses;
	GetRegisteredLayerClasses( layerClasses );

	for( int i = 0; i < layerClasses.Size(); ++i ) {
		// skip ignored layers
		if( ignoredLayers.Has( layerClasses[i] ) ) {
			continue;
		}
		// try to open file (it must exist)
		try {
			CArchiveFile file(
				GetTestDataFilePath( "data/LayersSerializationTestData", CString( layerClasses[i] ) + ".arch" ),
				CArchive::load );
		} catch( ... ) {
			GTEST_FAIL() << "archive is missing for " << CString( layerClasses[i] );
		}
	}
}

// ====================================================================================================================

#ifdef GENERATE_SERIALIZATION_FILES

static CPtr<CDnnBlob> generateBlob( int batchWidth, int imageHeight, int imageWidth, int imageDepth, int channelsCount )
{
	auto blob = CDnnBlob::Create3DImageBlob( MathEngine(), CT_Float, 1, batchWidth, imageHeight, imageWidth, imageDepth, channelsCount );
	CArray<float> buff;
	buff.Add( TestFloatValue, batchWidth * imageHeight * imageWidth * imageDepth * channelsCount );
	blob->CopyFrom( buff.GetPtr() );
	return blob;
}

static void setBaseParams( CBaseLayer& layer )
{
	layer.SetBaseLearningRate( TestFloatValue );
	layer.EnableLearning();
	layer.SetBaseL1RegularizationMult( 2 * TestFloatValue );
	layer.SetBaseL2RegularizationMult( 3 * TestFloatValue );
}

static void setSpecificParams( CBaseLayer& )
{
}

template <typename T>
static void serializeToFile( const CString& layerName )
{
	CRandom random;
	CDnn cnn( random, MathEngine() );

	CPtr<T> layerPtr = new T( MathEngine() );
	setBaseParams( *layerPtr );
	setSpecificParams( *layerPtr );
	layerPtr->SetName( LayerName );
	cnn.AddLayer( *layerPtr );

	CArchiveFile file( getFileName( layerName ), CArchive::SD_Storing );
	CArchive archive( &file, CArchive::SD_Storing );
	archive.Serialize( cnn );
}

GTEST_TEST( SerializeToFile, BaseLayerSerialization )
{
	serializeToFile<CSourceLayer>( "FmlCnnSourceLayer" );
	serializeToFile<CConcatChannelsLayer>( "FmlCnnConcatChannelsLayer" );
	serializeToFile<CConcatDepthLayer>( "FmlCnnConcatDepthLayer" );
	serializeToFile<CConcatHeightLayer>( "FmlCnnConcatHeightLayer" );
	serializeToFile<CConcatWidthLayer>( "FmlCnnConcatWidthLayer" );
	serializeToFile<CConcatBatchLengthLayer>( "FmlCnnConcatBatchLengthLayer" );
	serializeToFile<CConcatBatchWidthLayer>( "FmlCnnConcatBatchWidthLayer" );
	serializeToFile<CConcatListSizeLayer>( "FmlCnnConcatListSizeLayer" );
	serializeToFile<CConcatObjectLayer>( "FmlCnnConcatObjectLayer" );
	serializeToFile<CEltwiseSumLayer>( "FmlCnnEltwiseSumLayer" );
	serializeToFile<CEltwiseSubLayer>( "NeoMLDnnEltwiseSubLayer" );
	serializeToFile<CEltwiseMulLayer>( "FmlCnnEltwiseMulLayer" );
	serializeToFile<CEltwiseDivLayer>( "NeoMLDnnEltwiseDivLayer" );
	serializeToFile<CEltwiseNegMulLayer>( "FmlCnnEltwiseNegMulLayer" );
	serializeToFile<CEltwiseMaxLayer>( "FmlCnnEltwiseMaxLayer" );
	serializeToFile<CAbsLayer>( "FmlCnnAbsLayer" );
	serializeToFile<CSigmoidLayer>( "FmlCnnSigmoidLayer" );
	serializeToFile<CTanhLayer>( "FmlCnnTanhLayer" );
	serializeToFile<CHardTanhLayer>( "FmlCnnHardTanhLayer" );
	serializeToFile<CPowerLayer>( "FmlCnnPowerLayer" );
	serializeToFile<CCompositeSourceLayer>( "FmlCnnCompositeSourceLayer" );
	serializeToFile<CCompositeSinkLayer>( "FmlCompositeCnnSinkLayer" );
	serializeToFile<CAttentionWeightedSumLayer>( "FmlCnnAttentionWeightedSumLayer" );
	serializeToFile<CAttentionDotProductLayer>( "FmlCnnAttentionDotProductLayer" );
	serializeToFile<CAttentionSumLayer>( "FmlCnnAttentionSumLayer" );
	serializeToFile<CSequenceSumLayer>( "FmlCnnSequenceSumLayer" );
	serializeToFile<CBestSequenceLayer>( "FmlCnnBestSequenceLayer" );
	serializeToFile<CImageToPixelLayer>( "FmlCnnImageToPixelLayerClass" );
	serializeToFile<CCaptureSinkLayer>( "FmlCnnCaptureSink" );
	serializeToFile<CSinkLayer>( "FmlCnnSinkLayer" );
	serializeToFile<CAccumulativeLookupLayer>( "FmlCnnAccumulativeLookupLayer" );
	serializeToFile<CDotProductLayer>( "FmlCnnDotProductLayer" );
	serializeToFile<CHSwishLayer>( "FmlCnnHSwishLayer" );
	serializeToFile<CMatrixMultiplicationLayer>( "NeoMLDnnMatrixMultiplicationLayer" );
	serializeToFile<CAddToObjectLayer>( "NeoMLDnnAddToObjectLayer" );
	serializeToFile<CGELULayer>( "NeoMLDnnGELULayer" );
	serializeToFile<CGlobalMeanPoolingLayer>( "FmlCnnGlobalAveragePoolingLayer" );
	serializeToFile<CDataLayer>( "NeoMLDnnDataLayer" );
	serializeToFile<CBertConvLayer>( "NeoMLDnnBertConvLayer" );
	serializeToFile<CBroadcastLayer>( "NeoMLDnnBroadcastLayer" );
	serializeToFile<CExpLayer>( "NeoMLDnnExpLayer" );
	serializeToFile<CLogLayer>( "NeoMLDnnLogLayer" );
	serializeToFile<CNotLayer>( "NeoMLDnnNotLayer" );
	serializeToFile<CErfLayer>( "NeoMLDnnErfLayer" );
	serializeToFile<CLessLayer>( "NeoMLDnnLessLayer" );
	serializeToFile<CEqualLayer>( "NeoMLDnnEqualLayer" );
	serializeToFile<CGlobalSumPoolingLayer>( "NeoMLDnnGlobalSumPoolingLayer" );
	serializeToFile<CWhereLayer>( "NeoMLDnnWhereLayer" );
	serializeToFile<CScatterNDLayer>( "NeoMLDnnScatterNDLayer" );
}

#endif // GENERATE_SERIALIZATION_FILES

static void checkBlob( const CDnnBlob& blob, int expectedSize )
{
	CArray<float> buff;
	buff.SetSize( blob.GetDataSize() );
	blob.CopyTo( buff.GetPtr() );

	ASSERT_EQ( buff.Size(), expectedSize );
	for( int i = 0; i < expectedSize; ++i ) {
		EXPECT_NEAR( buff[i], TestFloatValue, 1e-3 );
	}
}

static void checkBaseParams( const CBaseLayer& layer )
{
	ASSERT_NEAR( layer.GetBaseLearningRate(), TestFloatValue, 1e-3 );
	ASSERT_NEAR( layer.GetBaseL1RegularizationMult(), 2 * TestFloatValue, 1e-3 );
	ASSERT_NEAR( layer.GetBaseL2RegularizationMult(), 3 * TestFloatValue, 1e-3 );
	ASSERT_EQ( layer.IsLearningEnabled(), static_cast<bool>( TestIntValue ) );
	ASSERT_EQ( layer.GetName(), LayerName );
}

template <typename T>
inline void checkSpecificParams( T& )
{
}

template <typename T>
static void checkSerializeFromFile( CDnn& cnn, const CString& fileName )
{
	CArchiveFile file( fileName, CArchive::SD_Loading );
	CArchive archive( &file, CArchive::SD_Loading );
	archive.Serialize( cnn );
	CPtr<CBaseLayer> layerPtr = cnn.GetLayer( LayerName );
	checkBaseParams( *layerPtr );
	checkSpecificParams( static_cast< T& >( *layerPtr ) );
}

template <typename T>
static void checkSerializeLayer( const CString& layerName )
{
	CRandom random;
	CDnn cnn( random, MathEngine() );
	checkSerializeFromFile<T>( cnn, getFileName( layerName ) );

	CString newVersionFileName = getFileName( layerName ) + ".new_ver";
	CArchiveFile file( newVersionFileName, CArchive::SD_Storing );
	CArchive archive( &file, CArchive::SD_Storing );
	archive.Serialize( cnn );
	archive.Close();
	file.Close();

	cnn.DeleteAllLayers();
	checkSerializeFromFile<T>( cnn, newVersionFileName );
}

GTEST_TEST( SerializeFromFile, BaseLayerSerialization )
{
	checkSerializeLayer<CBaseLayer>( "FmlCnnSourceLayer" );
	checkSerializeLayer<CBaseLayer>( "FmlCnnConcatChannelsLayer" );
	checkSerializeLayer<CBaseLayer>( "FmlCnnConcatDepthLayer" );
	checkSerializeLayer<CBaseLayer>( "FmlCnnConcatHeightLayer" );
	checkSerializeLayer<CBaseLayer>( "FmlCnnConcatWidthLayer" );
	checkSerializeLayer<CBaseLayer>( "FmlCnnConcatBatchLengthLayer" );
	checkSerializeLayer<CBaseLayer>( "FmlCnnConcatBatchWidthLayer" );
	checkSerializeLayer<CBaseLayer>( "FmlCnnConcatListSizeLayer" );
	checkSerializeLayer<CBaseLayer>( "FmlCnnConcatObjectLayer" );
	checkSerializeLayer<CBaseLayer>( "FmlCnnEltwiseSumLayer" );
	checkSerializeLayer<CBaseLayer>( "NeoMLDnnEltwiseSubLayer" );
	checkSerializeLayer<CBaseLayer>( "FmlCnnEltwiseMulLayer" );
	checkSerializeLayer<CBaseLayer>( "NeoMLDnnEltwiseDivLayer" );
	checkSerializeLayer<CBaseLayer>( "FmlCnnEltwiseNegMulLayer" );
	checkSerializeLayer<CBaseLayer>( "FmlCnnEltwiseMaxLayer" );
	checkSerializeLayer<CBaseLayer>( "FmlCnnAbsLayer" );
	checkSerializeLayer<CBaseLayer>( "FmlCnnSigmoidLayer" );
	checkSerializeLayer<CBaseLayer>( "FmlCnnTanhLayer" );
	checkSerializeLayer<CBaseLayer>( "FmlCnnHardTanhLayer" );
	checkSerializeLayer<CBaseLayer>( "FmlCnnPowerLayer" );
	checkSerializeLayer<CBaseLayer>( "FmlCnnCompositeSourceLayer" );
	checkSerializeLayer<CBaseLayer>( "FmlCompositeCnnSinkLayer" );
	checkSerializeLayer<CBaseLayer>( "FmlCnnAttentionWeightedSumLayer" );
	checkSerializeLayer<CBaseLayer>( "FmlCnnAttentionDotProductLayer" );
	checkSerializeLayer<CBaseLayer>( "FmlCnnAttentionSumLayer" );
	checkSerializeLayer<CBaseLayer>( "FmlCnnSequenceSumLayer" );
	checkSerializeLayer<CBaseLayer>( "FmlCnnBestSequenceLayer" );
	checkSerializeLayer<CBaseLayer>( "FmlCnnImageToPixelLayerClass" );
	checkSerializeLayer<CBaseLayer>( "FmlCnnCaptureSink" );
	checkSerializeLayer<CBaseLayer>( "FmlCnnSinkLayer" );
	checkSerializeLayer<CBaseLayer>( "FmlCnnAccumulativeLookupLayer" );
	checkSerializeLayer<CBaseLayer>( "FmlCnnDotProductLayer" );
	checkSerializeLayer<CBaseLayer>( "FmlCnnHSwishLayer" );
	checkSerializeLayer<CBaseLayer>( "NeoMLDnnMatrixMultiplicationLayer" );
	checkSerializeLayer<CBaseLayer>( "NeoMLDnnAddToObjectLayer" );
	checkSerializeLayer<CBaseLayer>( "NeoMLDnnGELULayer" );
	checkSerializeLayer<CBaseLayer>( "FmlCnnGlobalAveragePoolingLayer" );
	checkSerializeLayer<CBaseLayer>( "NeoMLDnnDataLayer" );
	checkSerializeLayer<CBaseLayer>( "NeoMLDnnBertConvLayer" );
	checkSerializeLayer<CBaseLayer>( "NeoMLDnnBroadcastLayer" );
	checkSerializeLayer<CBaseLayer>( "NeoMLDnnExpLayer" );
	checkSerializeLayer<CBaseLayer>( "NeoMLDnnLogLayer" );
	checkSerializeLayer<CBaseLayer>( "NeoMLDnnNotLayer" );
	checkSerializeLayer<CBaseLayer>( "NeoMLDnnErfLayer" );
	checkSerializeLayer<CBaseLayer>( "NeoMLDnnLessLayer" );
	checkSerializeLayer<CBaseLayer>( "NeoMLDnnEqualLayer" );
	checkSerializeLayer<CBaseLayer>( "NeoMLDnnGlobalSumPoolingLayer" );
	checkSerializeLayer<CBaseLayer>( "NeoMLDnnWhereLayer" );
	checkSerializeLayer<CBaseLayer>( "NeoMLDnnScatterNDLayer" );
}

// ====================================================================================================================

// CPoolingLayer

#ifdef GENERATE_SERIALIZATION_FILES

static void setSpecificParams( CPoolingLayer& layer )
{
	layer.SetStrideWidth( TestIntValue );
	layer.SetFilterWidth( 2 * TestIntValue );
	layer.SetFilterHeight( 3 * TestIntValue );
}

GTEST_TEST( SerializeToFile, PoolingLayerSerialization )
{
	serializeToFile<CMaxPoolingLayer>( "FmlCnnMaxPoolingLayer" );
	serializeToFile<CMeanPoolingLayer>( "FmlCnnMeanPoolingLayer" );
}

#endif // GENERATE_SERIALIZATION_FILES

template<>
inline void checkSpecificParams<CPoolingLayer>( CPoolingLayer& layer )
{
	ASSERT_EQ( layer.GetStrideWidth(), TestIntValue );
	ASSERT_EQ( layer.GetFilterWidth(), 2 * TestIntValue );
	ASSERT_EQ( layer.GetFilterHeight(), 3 * TestIntValue );
}

GTEST_TEST( SerializeFromFile, PoolingLayerSerialization )
{
	checkSerializeLayer<CPoolingLayer>( "FmlCnnMaxPoolingLayer" );
	checkSerializeLayer<CPoolingLayer>( "FmlCnnMeanPoolingLayer" );
}

// ====================================================================================================================

// C3dPoolingLayer

#ifdef GENERATE_SERIALIZATION_FILES

static void setSpecificParams( C3dPoolingLayer& layer )
{
	layer.SetFilterWidth( TestIntValue );
	layer.SetFilterHeight( 2 * TestIntValue );
	layer.SetStrideDepth( 3 * TestIntValue );
}

GTEST_TEST( SerializeToFile, 3dPoolingLayerSerialization )
{
	serializeToFile<C3dMaxPoolingLayer>( "FmlCnn3dMaxPoolingLayer" );
	serializeToFile<C3dMeanPoolingLayer>( "FmlCnn3dMeanPoolingLayer" );
}

#endif // GENERATE_SERIALIZATION_FILES

template<>
inline void checkSpecificParams<C3dPoolingLayer>( C3dPoolingLayer& layer )
{
	ASSERT_EQ( layer.GetFilterWidth(), TestIntValue );
	ASSERT_EQ( layer.GetFilterHeight(), 2 * TestIntValue );
	ASSERT_EQ( layer.GetStrideDepth(), 3 * TestIntValue );
}

GTEST_TEST( SerializeFromFile, 3dPoolingLayerSerialization )
{
	checkSerializeLayer<C3dMaxPoolingLayer>( "FmlCnn3dMaxPoolingLayer" );
	checkSerializeLayer<C3dMeanPoolingLayer>( "FmlCnn3dMeanPoolingLayer" );
}

// ====================================================================================================================

// CBaseConvLayer

#ifdef GENERATE_SERIALIZATION_FILES

static void setSpecificParams( CBaseConvLayer& layer )
{
	layer.SetFilterWidth( TestIntValue );
	layer.SetStrideWidth( TestIntValue );
	layer.SetPaddingHeight( TestIntValue );
	layer.SetFilterData( generateBlob( TestSize, 1, 1, 1, 1 ) );
	layer.SetFreeTermData( generateBlob( 1, 1, 1, 1, TestSize ) );
}

GTEST_TEST( SerializeToFile, BaseConvLayerSerialization )
{
	serializeToFile<CConvLayer>( "FmlCnnConvLayer" );
	serializeToFile<C3dConvLayer>( "FmlCnn3dConvLayer" );
	serializeToFile<CTransposedConvLayer>( "FmlCnnTransposedConvLayer" );
	serializeToFile<C3dTransposedConvLayer>( "FmlCnn3dTransposedConvLayer" );
	serializeToFile<CChannelwiseConvLayer>( "FmlCnnChannelwiseConvLayer" );
}

#endif // GENERATE_SERIALIZATION_FILES

template<>
inline void checkSpecificParams<CBaseConvLayer>( CBaseConvLayer& layer )
{
	ASSERT_EQ( layer.GetFilterWidth(), TestIntValue );
	ASSERT_EQ( layer.GetStrideWidth(), TestIntValue );
	ASSERT_EQ( layer.GetPaddingHeight(), TestIntValue );
	checkBlob( *layer.GetFreeTermData(), TestSize );
	checkBlob( *layer.GetFilterData(), TestSize );
}

GTEST_TEST( SerializeFromFile, BaseConvLayerSerialization )
{
	checkSerializeLayer<CConvLayer>( "FmlCnnConvLayer" );
	checkSerializeLayer<C3dConvLayer>( "FmlCnn3dConvLayer" );
	checkSerializeLayer<CTransposedConvLayer>( "FmlCnnTransposedConvLayer" );
	checkSerializeLayer<C3dTransposedConvLayer>( "FmlCnn3dTransposedConvLayer" );
	checkSerializeLayer<CChannelwiseConvLayer>( "FmlCnnChannelwiseConvLayer" );
}

// ====================================================================================================================

// CBatchNormalizationLayer

#ifdef GENERATE_SERIALIZATION_FILES

static void setSpecificParams( CBatchNormalizationLayer& layer )
{
	auto blob = generateBlob( 1, TestSize, 1, 1, 1 );
	layer.SetFinalParams( blob );
	layer.SetSlowConvergenceRate( 0.5 );
	layer.SetZeroFreeTerm( TestIntValue );
}

GTEST_TEST( SerializeToFile, BatchNormalizationLayerSerialization )
{
	serializeToFile<CBatchNormalizationLayer>( "FmlCnnBatchNormalizationLayer" );
}

#endif // GENERATE_SERIALIZATION_FILES

template<>
inline void checkSpecificParams<CBatchNormalizationLayer>( CBatchNormalizationLayer& layer )
{
	auto blob = layer.GetFinalParams();
	checkBlob( *blob, TestSize );
	ASSERT_NEAR( layer.GetSlowConvergenceRate(), 0.5, 1e-3 );
	ASSERT_EQ( layer.IsZeroFreeTerm(), static_cast<bool>( TestIntValue ) );
}

GTEST_TEST( SerializeFromFile, BatchNormalizationLayerSerialization )
{
	checkSerializeLayer<CBatchNormalizationLayer>( "FmlCnnBatchNormalizationLayer" );
}

// ====================================================================================================================

// CLinearLayer

#ifdef GENERATE_SERIALIZATION_FILES

static void setSpecificParams( CLinearLayer& layer )
{
	layer.SetMultiplier( TestFloatValue );
	layer.SetFreeTerm( TestFloatValue );
}

GTEST_TEST( SerializeToFile, LinearLayerSerialization )
{
	serializeToFile<CLinearLayer>( "FmlCnnLinearLayer" );
}

#endif // GENERATE_SERIALIZATION_FILES

template<>
inline void checkSpecificParams<CLinearLayer>( CLinearLayer& layer )
{
	ASSERT_NEAR( layer.GetMultiplier(), TestFloatValue, 1e-3 );
	ASSERT_NEAR( layer.GetFreeTerm(), TestFloatValue, 1e-3 );
}

GTEST_TEST( SerializeFromFile, LinearLayerSerialization )
{
	checkSerializeLayer<CLinearLayer>( "FmlCnnLinearLayer" );
}

// ====================================================================================================================

// CBaseSplitLayer

#ifdef GENERATE_SERIALIZATION_FILES

static void setSpecificParams( CBaseSplitLayer& layer )
{
	CArray<int> buf;
	buf.Add( TestIntValue, TestSize );
	layer.SetOutputCounts( buf );
}

GTEST_TEST( SerializeToFile, BaseSplitSerialization )
{
	serializeToFile<CSplitChannelsLayer>( "FmlCnnSplitChannelsLayer" );
	serializeToFile<CSplitDepthLayer>( "FmlCnnSplitDepthLayer" );
	serializeToFile<CSplitWidthLayer>( "FmlCnnSplitWidthLayer" );
	serializeToFile<CSplitHeightLayer>( "FmlCnnSplitHeightLayer" );
	serializeToFile<CSplitListSizeLayer>( "NeoMLDnnSplitListSizeLayer" );
	serializeToFile<CSplitBatchWidthLayer>( "FmlCnnSplitBatchWidthLayer" );
	serializeToFile<CSplitBatchLengthLayer>( "NeoMLDnnSplitBatchLengthLayer" );
}

#endif // GENERATE_SERIALIZATION_FILES

template<>
inline void checkSpecificParams<CBaseSplitLayer>( CBaseSplitLayer& layer )
{
	CArray<int> buf;
	layer.GetOutputCounts().CopyTo( buf );
	ASSERT_EQ( buf.Size(), TestSize );
	for( int i = 0; i < buf.Size(); ++i) {
		ASSERT_EQ( buf[i], TestIntValue );
	}
}

GTEST_TEST( SerializeFromFile, BaseSplitSerialization )
{
	checkSerializeLayer<CBaseSplitLayer>( "FmlCnnSplitChannelsLayer" );
	checkSerializeLayer<CBaseSplitLayer>( "FmlCnnSplitDepthLayer" );
	checkSerializeLayer<CBaseSplitLayer>( "FmlCnnSplitWidthLayer" );
	checkSerializeLayer<CBaseSplitLayer>( "FmlCnnSplitHeightLayer" );
	checkSerializeLayer<CBaseSplitLayer>( "NeoMLDnnSplitListSizeLayer" );
	checkSerializeLayer<CBaseSplitLayer>( "FmlCnnSplitBatchWidthLayer" );
	checkSerializeLayer<CBaseSplitLayer>( "NeoMLDnnSplitBatchLengthLayer" );
}

// ====================================================================================================================

// CELULayer

#ifdef GENERATE_SERIALIZATION_FILES

static void setSpecificParams( CELULayer& layer )
{
	layer.SetAlpha( TestFloatValue );
}

GTEST_TEST( SerializeToFile, ELULayerSerialization )
{
	serializeToFile<CELULayer>( "FmlCnnELULayer" );
	serializeToFile<CLeakyReLULayer>( "FmlCnnLeakyReLULayer" );
}

#endif // GENERATE_SERIALIZATION_FILES

template<>
inline void checkSpecificParams<CELULayer>( CELULayer& layer )
{
	ASSERT_NEAR( layer.GetAlpha(), TestFloatValue, 1e-3 );
}

GTEST_TEST( SerializeFromFile, ELULayerSerialization )
{
	checkSerializeLayer<CELULayer>( "FmlCnnELULayer" );
	checkSerializeLayer<CLeakyReLULayer>( "FmlCnnLeakyReLULayer" );
}

// ====================================================================================================================

// CHardSigmoidLayer

#ifdef GENERATE_SERIALIZATION_FILES

static void setSpecificParams( CHardSigmoidLayer& layer )
{
	layer.SetSlope( 0.3f );
	layer.SetBias( 0.4f );
}

GTEST_TEST( SerializeToFile, HardSigmoidLayerSerialization )
{
	serializeToFile<CHardSigmoidLayer>( "FmlCnnSigmoidTanhLayer" );
}

#endif // GENERATE_SERIALIZATION_FILES

template<>
inline void checkSpecificParams<CHardSigmoidLayer>( CHardSigmoidLayer& layer )
{
	ASSERT_NEAR( layer.GetSlope(), 0.3f, 1e-3 );
	ASSERT_NEAR( layer.GetBias(), 0.4f, 1e-3 );
}

GTEST_TEST( SerializeFromFile, HardSigmoidLayerSerialization )
{
	checkSerializeLayer<CHardSigmoidLayer>( "FmlCnnSigmoidTanhLayer" );
}

// ====================================================================================================================

// CReLULayer

#ifdef GENERATE_SERIALIZATION_FILES

static void setSpecificParams( CReLULayer& layer )
{
	layer.SetUpperThreshold( TestFloatValue );
}

GTEST_TEST( SerializeToFile, ReLULayerSerialization )
{
	serializeToFile<CReLULayer>( "FmlCnnReLULayer" );
}

#endif // GENERATE_SERIALIZATION_FILES

template<>
inline void checkSpecificParams<CReLULayer>( CReLULayer& layer )
{
	ASSERT_NEAR( layer.GetUpperThreshold(), TestFloatValue, 1e-3 );
}

GTEST_TEST( SerializeFromFile, ReLULayerSerialization )
{
	checkSerializeLayer<CReLULayer>( "FmlCnnReLULayer" );
}

// ====================================================================================================================

// CRleConvLayer

#ifdef GENERATE_SERIALIZATION_FILES

static void setSpecificParams( CRleConvLayer& layer )
{
	layer.SetStrokeValue( TestFloatValue );
	layer.SetNonStrokeValue( TestFloatValue );
	layer.SetFilterData( generateBlob( 1, TestSize, 1, 1, 1 ) );
	layer.SetFilterHeight( TestSize );
}

GTEST_TEST( SerializeToFile, RleConvLayerSerialization )
{
	serializeToFile<CRleConvLayer>( "FmlCnnRleConvLayer" );
}

#endif // GENERATE_SERIALIZATION_FILES

template<>
inline void checkSpecificParams<CRleConvLayer>( CRleConvLayer& layer )
{
	ASSERT_NEAR( layer.GetStrokeValue(), TestFloatValue, 1e-3 );
	ASSERT_NEAR( layer.GetNonStrokeValue(), TestFloatValue, 1e-3 );
	checkBlob( *layer.GetFilterData(), TestSize );
	ASSERT_EQ( layer.GetFilterHeight(), TestSize );
}

GTEST_TEST( SerializeFromFile, RleConvLayerSerialization )
{
	checkSerializeLayer<CRleConvLayer>( "FmlCnnRleConvLayer" );
}

// ====================================================================================================================

// CFullyConnectedLayer

#ifdef GENERATE_SERIALIZATION_FILES

static void setSpecificParams( CFullyConnectedLayer& layer )
{
	layer.SetNumberOfElements( TestSize );
	auto blob = generateBlob( 1, 1, 1, 1, TestSize );
	layer.SetWeightsData( blob );
	layer.SetFreeTermData( blob );
}

GTEST_TEST( SerializeToFile, FullyConnectedLayerSerialization )
{
	serializeToFile<CFullyConnectedLayer>( "FmlCnnFullyConnectedLayer" );
}

#endif // GENERATE_SERIALIZATION_FILES

template<>
inline void checkSpecificParams<CFullyConnectedLayer>( CFullyConnectedLayer& layer )
{
	ASSERT_EQ( layer.GetNumberOfElements(), TestSize );
	checkBlob( *layer.GetWeightsData(), TestSize );
	checkBlob( *layer.GetWeightsData(), TestSize );
}

GTEST_TEST( SerializeFromFile, FullyConnectedLayerSerialization )
{
	checkSerializeLayer<CFullyConnectedLayer>( "FmlCnnFullyConnectedLayer" );
}

// ====================================================================================================================

// CFullyConnectedSourceLayer

#ifdef GENERATE_SERIALIZATION_FILES

static void setSpecificParams( CFullyConnectedSourceLayer& layer )
{
	layer.SetBatchSize( TestSize );
	layer.SetMaxBatchCount( TestSize );
}

GTEST_TEST( SerializeToFile, FullyConnectedSourceLayerSerialization )
{
	serializeToFile<CFullyConnectedSourceLayer>( "FmlCnnFullyConnectedSourceLayer" );
}

#endif // GENERATE_SERIALIZATION_FILES

template<>
inline void checkSpecificParams<CFullyConnectedSourceLayer>( CFullyConnectedSourceLayer& layer )
{
	ASSERT_EQ( layer.GetBatchSize(), TestSize );
	ASSERT_EQ( layer.GetMaxBatchCount(), TestSize );
}

GTEST_TEST( SerializeFromFile, FullyConnectedSourceLayerSerialization )
{
	checkSerializeLayer<CFullyConnectedSourceLayer>( "FmlCnnFullyConnectedSourceLayer" );
}

// ====================================================================================================================

// CLossLayer

#ifdef GENERATE_SERIALIZATION_FILES

static void setSpecificParams( CLossLayer& layer )
{
	layer.SetLossWeight( 0.5 );
}

GTEST_TEST( SerializeToFile, LossLayerSerialization )
{
	serializeToFile<CEuclideanLossLayer>( "FmlCnnEuclideanLossLayer" );
	serializeToFile<CHingeLossLayer>( "FmlCnnHingeLossLayer" );
	serializeToFile<CSquaredHingeLossLayer>( "FmlCnnSquaredHingeLossLayer" );
	serializeToFile<CCrfInternalLossLayer>( "FmlCnnCrfInternalLossLayer" );
	serializeToFile<CMultiHingeLossLayer>( "FmlCnnMultyHingeLossLayer" );
	serializeToFile<CMultiSquaredHingeLossLayer>( "FmlCnnMultySquaredHingeLossLayer" );
	serializeToFile<CL1LossLayer>( "NeoMLDnnL1LossLayer" );
}

#endif

template<>
inline void checkSpecificParams<CLossLayer>( CLossLayer& layer )
{
	ASSERT_NEAR( layer.GetLossWeight(), 0.5, 1e-3 );
}

GTEST_TEST( SerializeFromFile, LossLayerSerialization )
{
	checkSerializeLayer<CLossLayer>( "FmlCnnEuclideanLossLayer" );
	checkSerializeLayer<CLossLayer>( "FmlCnnHingeLossLayer" );
	checkSerializeLayer<CLossLayer>( "FmlCnnSquaredHingeLossLayer" );
	checkSerializeLayer<CLossLayer>( "FmlCnnCrfInternalLossLayer" );
	checkSerializeLayer<CLossLayer>( "FmlCnnMultyHingeLossLayer" );
	checkSerializeLayer<CLossLayer>( "FmlCnnMultySquaredHingeLossLayer" );
	checkSerializeLayer<CLossLayer>( "NeoMLDnnL1LossLayer" );
}

// ====================================================================================================================

// CCrossEntropyLossLayer

template<>
inline void checkSpecificParams<CCrossEntropyLossLayer>( CCrossEntropyLossLayer& layer )
{
	ASSERT_EQ( layer.IsSoftmaxApplied(), static_cast<bool>( TestIntValue ) );
}

#ifdef GENERATE_SERIALIZATION_FILES

static void setSpecificParams( CCrossEntropyLossLayer& layer )
{
	layer.SetApplySoftmax( TestIntValue );
}

GTEST_TEST( SerializeToFile, CrossEntropyLossLayerSerialization )
{
	serializeToFile<CCrossEntropyLossLayer>( "FmlCnnCrossEntropyLossLayer" );
}

#endif

GTEST_TEST( SerializeFromFile, CrossEntropyLossLayerSerialization )
{
	checkSerializeLayer<CCrossEntropyLossLayer>( "FmlCnnCrossEntropyLossLayer" );
}

// ====================================================================================================================

// CBinaryCrossEntropyLossLayer

template<>
inline void checkSpecificParams<CBinaryCrossEntropyLossLayer>( CBinaryCrossEntropyLossLayer& layer )
{
	ASSERT_NEAR( layer.GetPositiveWeight(), TestFloatValue, 1e-3 );
}

#ifdef GENERATE_SERIALIZATION_FILES

static void setSpecificParams( CBinaryCrossEntropyLossLayer& layer )
{
	layer.SetPositiveWeight( TestFloatValue );
}

GTEST_TEST( SerializeToFile, BinaryCrossEntropyLossLayerSerialization )
{
	serializeToFile<CBinaryCrossEntropyLossLayer>( "FmlCnnBinaryCrossEntropyLossLayer" );
}

#endif

GTEST_TEST( SerializeFromFile, BinaryCrossEntropyLossLayerSerialization )
{
	checkSerializeLayer<CBinaryCrossEntropyLossLayer>( "FmlCnnBinaryCrossEntropyLossLayer" );
}

// ====================================================================================================================

// CProblemSourceLayer

template<>
inline void checkSpecificParams<CProblemSourceLayer>( CProblemSourceLayer& layer )
{
	ASSERT_EQ( TestIntValue, layer.GetBatchSize() );
	ASSERT_EQ( TBlobType::CT_Int, layer.GetLabelType() );
}

#ifdef GENERATE_SERIALIZATION_FILES

static void setSpecificParams( CProblemSourceLayer& layer )
{
	layer.SetBatchSize( TestIntValue );
	layer.SetLabelType( TBlobType::CT_Int );
}

GTEST_TEST( SerializeToFile, ProblemSourceLayerSerialization )
{
	serializeToFile<CProblemSourceLayer>( "FmlCnnProblemSourceLayer" );
}

#endif

GTEST_TEST( SerializeFromFile, ProblemSourceLayerSerialization )
{
	checkSerializeLayer<CProblemSourceLayer>( "FmlCnnProblemSourceLayer" );
}

// ====================================================================================================================

// CDropoutLayer

template<>
inline void checkSpecificParams<CDropoutLayer>( CDropoutLayer& layer )
{
	ASSERT_NEAR( layer.GetDropoutRate(), 0.5, 1e-3 );
	ASSERT_EQ( true, layer.IsSpatial() );
	ASSERT_EQ( true, layer.IsBatchwise() );
}

#ifdef GENERATE_SERIALIZATION_FILES

static void setSpecificParams( CDropoutLayer& layer )
{
	layer.SetDropoutRate( 0.5 );
	layer.SetSpatial( true );
	layer.SetBatchwise( true );
}

GTEST_TEST( SerializeToFile, DropoutLayerSerialization )
{
	serializeToFile<CDropoutLayer>( "FmlCnnDropoutLayer" );
}

#endif

GTEST_TEST( SerializeFromFile, DropoutLayerSerialization )
{
	checkSerializeLayer<CDropoutLayer>( "FmlCnnDropoutLayer" );
}

// ====================================================================================================================

// CImageResizeLayer

template<>
inline void checkSpecificParams<CImageResizeLayer>( CImageResizeLayer& layer )
{
	ASSERT_EQ( TestIntValue, layer.GetDelta( CImageResizeLayer::TImageSide::IS_Top ) );
	ASSERT_NEAR( TestFloatValue, layer.GetDefaultValue(), 1e-3 );
}

#ifdef GENERATE_SERIALIZATION_FILES

static void setSpecificParams( CImageResizeLayer& layer )
{
	layer.SetDelta( CImageResizeLayer::TImageSide::IS_Top, TestIntValue );
	layer.SetDefaultValue( TestFloatValue );
}

GTEST_TEST( SerializeToFile, ImageResizeLayerSerialization )
{
	serializeToFile<CImageResizeLayer>( "FmlCnnImageResizeLayer" );
}

#endif

GTEST_TEST( SerializeFromFile, ImageResizeLayerSerialization )
{
	checkSerializeLayer<CImageResizeLayer>( "FmlCnnImageResizeLayer" );
}

// ====================================================================================================================

// CDnnMultychannelLookupLayer

template<>
inline void checkSpecificParams<CMultichannelLookupLayer>( CMultichannelLookupLayer& layer )
{
	const CArray<CLookupDimension>& dimensions = layer.GetDimensions();
	ASSERT_EQ( 2, dimensions.Size() );
	ASSERT_EQ( dimensions[0].VectorCount, 3 );
	ASSERT_EQ( dimensions[0].VectorSize, 5 );
	ASSERT_EQ( dimensions[1].VectorCount, 2 );
	ASSERT_EQ( dimensions[1].VectorSize, 4 );
	checkBlob( *layer.GetEmbeddings( 0 ), 15 );
	checkBlob( *layer.GetEmbeddings( 1 ), 8 );
	ASSERT_EQ( true, layer.IsUseFrameworkLearning() );
}

#ifdef GENERATE_SERIALIZATION_FILES

static void setSpecificParams( CMultichannelLookupLayer& layer )
{
	CArray<CLookupDimension> dimensions = { { 3, 5 },{ 2, 4 } };
	layer.SetDimensions( dimensions );
	layer.SetEmbeddings( generateBlob( 3, 5, 1, 1, 1 ), 0 );
	layer.SetEmbeddings( generateBlob( 2, 4, 1, 1, 1 ), 1 );
	layer.SetUseFrameworkLearning( true );
}

GTEST_TEST( SerializeToFile, MultychannelLookupLayerSerialization )
{
	serializeToFile<CMultichannelLookupLayer>( "FmlCnnMultychannelLookupLayer" );
}

#endif

GTEST_TEST( SerializeFromFile, MultychannelLookupLayerSerialization )
{
	checkSerializeLayer<CMultichannelLookupLayer>( "FmlCnnMultychannelLookupLayer" );
}

// ====================================================================================================================

// CCompositeLayer

template<>
inline void checkSpecificParams<CCompositeLayer>( CCompositeLayer& layer )
{
	ASSERT_EQ( true, layer.AreInternalLogsEnabled() );
	CArray<const char*> layers;
	layer.GetLayerList( layers );
	ASSERT_EQ( 2, layers.Size() );
	ASSERT_EQ( "Layer0", CString( layers[0] ) );
	ASSERT_EQ( "Layer1", CString( layers[1] ) );
}

#ifdef GENERATE_SERIALIZATION_FILES

static void setSpecificParams( CCompositeLayer& layer )
{
	CPtr<CConvLayer> layer0Ptr = new CConvLayer( MathEngine() );
	layer0Ptr->SetName( "Layer0" );
	CPtr<CConvLayer> layer1Ptr = new CConvLayer( MathEngine() );
	layer1Ptr->SetName( "Layer1" );
	layer.AddLayer( *layer0Ptr );
	layer.AddLayer( *layer1Ptr );
	layer.SetInputMapping( 0, *layer0Ptr );
	layer.SetInputMapping( 1, *layer1Ptr );
	layer.EnableInternalLogging();
}

GTEST_TEST( SerializeToFile, CompositeLayerSerialization )
{
	serializeToFile<CCompositeLayer>( "FmlCnnCompositeLayer" );
	serializeToFile<CRecurrentLayer>( "FmlCnnRecurrentLayer" );
}

#endif

GTEST_TEST( SerializeFromFile, CompositeLayerSerialization )
{
	checkSerializeLayer<CCompositeLayer>( "FmlCnnCompositeLayer" );
	checkSerializeLayer<CCompositeLayer>( "FmlCnnRecurrentLayer" );
}

// ====================================================================================================================

// CSubSequenceLayer

template<>
inline void checkSpecificParams<CSubSequenceLayer>( CSubSequenceLayer& layer )
{
	ASSERT_EQ( 1, layer.GetStartPos() );
	ASSERT_EQ( 3, layer.GetLength() );
}

#ifdef GENERATE_SERIALIZATION_FILES

static void setSpecificParams( CSubSequenceLayer& layer )
{
	layer.SetStartPos( 1 );
	layer.SetLength( 3 );
}

GTEST_TEST( SerializeToFile, SubSequenceLayerSerialization )
{
	serializeToFile<CSubSequenceLayer>( "FmlCnnSubSequenceLayer" );
}

#endif

GTEST_TEST( SerializeFromFile, SubSequenceLayerSerialization )
{
	checkSerializeLayer<CSubSequenceLayer>( "FmlCnnSubSequenceLayer" );
}

// ====================================================================================================================

// CBackLinkLayer

template<>
inline void checkSpecificParams<CBackLinkLayer>( CBackLinkLayer& layer )
{
	ASSERT_EQ( 4, layer.GetDimSize( TBlobDim::BD_Height ) );
	ASSERT_EQ( 5, layer.GetDimSize( TBlobDim::BD_Width ) );
	ASSERT_NEAR( layer.CaptureSink()->GetBaseLearningRate(), 0.5, 1e-3 );
}

#ifdef GENERATE_SERIALIZATION_FILES

static void setSpecificParams( CBackLinkLayer& layer )
{
	layer.SetDimSize( TBlobDim::BD_Height, 4 );
	layer.SetDimSize( TBlobDim::BD_Width, 5 );
	layer.CaptureSink()->SetBaseLearningRate( 0.5 );
}

GTEST_TEST( SerializeToFile, BackLinkSerialization )
{
	serializeToFile<CBackLinkLayer>( "FmlBackLinkClassName" );
}

#endif

GTEST_TEST( SerializeFromFile, BackLinkSerialization )
{
	checkSerializeLayer<CBackLinkLayer>( "FmlCnnBackLink" );
}

// ====================================================================================================================

// CEnumBinarizationLayer

template<>
inline void checkSpecificParams<CEnumBinarizationLayer>( CEnumBinarizationLayer& layer )
{
	ASSERT_EQ( 5, layer.GetEnumSize() );
}

#ifdef GENERATE_SERIALIZATION_FILES

static void setSpecificParams( CEnumBinarizationLayer& layer )
{
	layer.SetEnumSize( 5 );
}

GTEST_TEST( SerializeToFile, EnumBinarizationLayerSerialization )
{
	serializeToFile<CEnumBinarizationLayer>( "FmlCnnEnumBinarizationLayer" );
}

#endif

GTEST_TEST( SerializeFromFile, EnumBinarizationLayerSerialization )
{
	checkSerializeLayer<CEnumBinarizationLayer>( "FmlCnnEnumBinarizationLayer" );
}

// ====================================================================================================================

// CBitSetVectorizationLayer

template<>
inline void checkSpecificParams<CBitSetVectorizationLayer>( CBitSetVectorizationLayer& layer )
{
	ASSERT_EQ( 5, layer.GetBitSetSize() );
}

#ifdef GENERATE_SERIALIZATION_FILES

static void setSpecificParams( CBitSetVectorizationLayer& layer )
{
	layer.SetBitSetSize( 5 );
}

GTEST_TEST( SerializeToFile, BitSetVectorizationLayerSerialization )
{
	serializeToFile<CBitSetVectorizationLayer>( "FmlCnnBitSetVectorizationLayerClassName" );
}

#endif

GTEST_TEST( SerializeFromFile, BitSetVectorizationLayerSerialization )
{
	checkSerializeLayer<CBitSetVectorizationLayer>( "FmlCnnBitSetVectorizationLayerClassName" );
}

// ====================================================================================================================

// CSoftmaxLayer

template<>
inline void checkSpecificParams<CSoftmaxLayer>( CSoftmaxLayer& layer )
{
	ASSERT_EQ( CSoftmaxLayer::TNormalizationArea::NA_ListSize, layer.GetNormalizationArea() );
}

#ifdef GENERATE_SERIALIZATION_FILES

static void setSpecificParams( CSoftmaxLayer& layer )
{
	layer.SetNormalizationArea( CSoftmaxLayer::TNormalizationArea::NA_ListSize );
}

GTEST_TEST( SerializeToFile, SoftmaxLayerSerialization )
{
	serializeToFile<CSoftmaxLayer>( "FmlCnnSoftmaxLayer" );
}

#endif

GTEST_TEST( SerializeFromFile, SoftmaxLayerSerialization )
{
	checkSerializeLayer<CSoftmaxLayer>( "FmlCnnSoftmaxLayer" );
}

// ====================================================================================================================

// CGlobalMaxPoolingLayer

template<>
inline void checkSpecificParams<CGlobalMaxPoolingLayer>( CGlobalMaxPoolingLayer& layer )
{
	ASSERT_EQ( 5, layer.GetMaxCount() );
}

#ifdef GENERATE_SERIALIZATION_FILES

static void setSpecificParams( CGlobalMaxPoolingLayer& layer )
{
	layer.SetMaxCount( 5 );
}

GTEST_TEST( SerializeToFile, GlobalMaxPoolingLayerSerialization )
{
	serializeToFile<CGlobalMaxPoolingLayer>( "FmlCnnGlobalMaxPoolingLayer" );
}

#endif

GTEST_TEST( SerializeFromFile, GlobalMaxPoolingLayerSerialization )
{
	checkSerializeLayer<CGlobalMaxPoolingLayer>( "FmlCnnGlobalMaxPoolingLayer" );
}

// ====================================================================================================================

// CLstmLayer

static CPtr<CDnnBlob> concatLstmWeights( CDnnBlob& inputWeights, CDnnBlob& recurWeights )
{
	// Reinterpret all of the blobs like BatchWidthxChannels.
	CArray<CBlobDesc> mergeDesc;
	CArray<CFloatHandle> mergeData;

	CBlobDesc weightDesc( CT_Float );
	weightDesc.SetDimSize( BD_BatchWidth, inputWeights.GetObjectCount() );
	weightDesc.SetDimSize( BD_Channels, inputWeights.GetObjectSize() );
	mergeDesc.Add( weightDesc );
	mergeData.Add( inputWeights.GetData() );

	weightDesc.SetDimSize( BD_Channels, recurWeights.GetObjectSize() );
	mergeDesc.Add( weightDesc );
	mergeData.Add( recurWeights.GetData() );

	weightDesc.SetDimSize( BD_Channels, inputWeights.GetObjectSize() + recurWeights.GetObjectSize() );
	CPtr<CDnnBlob> weight = CDnnBlob::CreateBlob( MathEngine(), weightDesc );

	MathEngine().BlobMergeByDim( BD_Channels, mergeDesc.GetPtr(), mergeData.GetPtr(), 2, weightDesc, weight->GetData() );

	return weight;
}

template<>
inline void checkSpecificParams<CLstmLayer>( CLstmLayer& layer )
{
	ASSERT_EQ( TActivationFunction::AF_LeakyReLU, layer.GetRecurrentActivation() );
	checkBlob( *concatLstmWeights( *layer.GetInputWeightsData(), *layer.GetRecurWeightsData() ), TestSize * TestSize );
	checkBlob( *layer.GetInputFreeTermData(), TestSize );
	ASSERT_EQ( layer.GetRecurFreeTermData(), nullptr );
}

#ifdef GENERATE_SERIALIZATION_FILES

static void setSpecificParams( CLstmLayer& layer )
{
	layer.SetRecurrentActivation( TActivationFunction::AF_LeakyReLU );
	layer.SetInputWeightsData( generateBlob( TestSize, 1, 1, 1, TestSize * 3 / 4 ) );
	layer.SetInputFreeTermData( generateBlob( 1, 1, 1, 1, TestSize ) );
	layer.SetRecurWeightsData( generateBlob( TestSize, 1, 1, 1, TestSize / 4 ) );
}

GTEST_TEST( SerializeToFile, LstmLayerSerialization )
{
	serializeToFile<CLstmLayer>( "FmlCnnLstmLayer" );
}

#endif

GTEST_TEST( SerializeFromFile, LstmLayerSerialization )
{
	checkSerializeLayer<CLstmLayer>( "FmlCnnLstmLayer" );
}

// ====================================================================================================================

// CGruLayer

template<>
inline void checkSpecificParams<CGruLayer>( CGruLayer& layer )
{
	checkBlob( *layer.GetMainWeightsData(), TestSize );
	checkBlob( *layer.GetMainFreeTermData(), TestSize );
	checkBlob( *layer.GetGateWeightsData(), TestSize );
	checkBlob( *layer.GetGateFreeTermData(), TestSize );
}

#ifdef GENERATE_SERIALIZATION_FILES

static void setSpecificParams( CGruLayer& layer )
{
	layer.SetMainWeightsData( generateBlob( 1, 1, 1, 1, TestSize ) );
	layer.SetMainFreeTermData( generateBlob( 1, 1, 1, 1, TestSize ) );
	layer.SetGateWeightsData( generateBlob( 1, 1, 1, 1, TestSize ) );
	layer.SetGateFreeTermData( generateBlob( 1, 1, 1, 1, TestSize ) );
}

GTEST_TEST( SerializeToFile, GruLayerSerialization )
{
	serializeToFile<CGruLayer>( "FmlCnnGruLayer" );
}

#endif

GTEST_TEST( SerializeFromFile, GruLayerSerialization )
{
	checkSerializeLayer<CGruLayer>( "FmlCnnGruLayer" );
}

// ====================================================================================================================

// CMaxOverTimePoolingLayer

template<>
inline void checkSpecificParams<CMaxOverTimePoolingLayer>( CMaxOverTimePoolingLayer& layer )
{
	ASSERT_EQ( 4, layer.GetFilterLength() );
	ASSERT_EQ( 5, layer.GetStrideLength() );
}

#ifdef GENERATE_SERIALIZATION_FILES

static void setSpecificParams( CMaxOverTimePoolingLayer& layer )
{
	layer.SetFilterLength( 4 );
	layer.SetStrideLength( 5 );
}

GTEST_TEST( SerializeToFile, MaxOverTimePoolingLayerSerialization )
{
	serializeToFile<CMaxOverTimePoolingLayer>( "FmlCnnMaxOverTimePoolingLayer" );
}

#endif

GTEST_TEST( SerializeFromFile, MaxOverTimePoolingLayerSerialization )
{
	checkSerializeLayer<CMaxOverTimePoolingLayer>( "FmlCnnMaxOverTimePoolingLayer" );
}

// ====================================================================================================================

// CTimeConvLayer

template<>
inline void checkSpecificParams<CTimeConvLayer>( CTimeConvLayer& layer )
{
	ASSERT_EQ( 4, layer.GetDilation() );
	ASSERT_EQ( 5, layer.GetPadding() );
	ASSERT_EQ( 6, layer.GetStride() );
}

#ifdef GENERATE_SERIALIZATION_FILES

static void setSpecificParams( CTimeConvLayer& layer )
{
	layer.SetDilation( 4 );
	layer.SetPadding( 5 );
	layer.SetStride( 6 );
	auto blob = generateBlob( 1, 1, 1, 1, TestSize );
	layer.SetFilterData( blob );
	layer.SetFreeTermData( blob );
}

GTEST_TEST( SerializeToFile, TimeConvLayerSerialization )
{
	serializeToFile<CTimeConvLayer>( "FmlCnnTimeConvLayer" );
}

#endif

GTEST_TEST( SerializeFromFile, TimeConvLayerSerialization )
{
	checkSerializeLayer<CTimeConvLayer>( "FmlCnnTimeConvLayer" );
}

// ====================================================================================================================

// CCrfLayer

template<>
inline void checkSpecificParams<CCrfLayer>( CCrfLayer& layer )
{
	ASSERT_EQ( 4, layer.GetNumberOfClasses() );
	ASSERT_EQ( 2, layer.GetPaddingClass() );
	ASSERT_NEAR( layer.GetDropoutRate(), 0.5, 1e-3 );
}

#ifdef GENERATE_SERIALIZATION_FILES

static void setSpecificParams( CCrfLayer& layer )
{
	layer.SetNumberOfClasses( 4 );
	layer.SetPaddingClass( 2 );
	layer.SetDropoutRate( 0.5 );
}

GTEST_TEST( SerializeToFile, CrfLayerSerialization )
{
	serializeToFile<CCrfLayer>( "FmlCnnCrfLayer" );
	serializeToFile<CCrfCalculationLayer>( "FmlCnnCrfCalculationLayer" );
}

#endif

GTEST_TEST( SerializeFromFile, CrfLayerSerialization )
{
	checkSerializeLayer<CCrfLayer>( "FmlCnnCrfLayer" );
	checkSerializeLayer<CCrfCalculationLayer>( "FmlCnnCrfCalculationLayer" );
}

// ====================================================================================================================

// CCrfLossLayer

template<>
inline void checkSpecificParams<CCrfLossLayer>( CCrfLossLayer& layer )
{
	ASSERT_NEAR( layer.GetLossWeight(), 0.5, 1e-3 );
}

#ifdef GENERATE_SERIALIZATION_FILES

static void setSpecificParams( CCrfLossLayer& layer )
{
	layer.SetLossWeight( 0.5 );
}

GTEST_TEST( SerializeToFile, CrfLossLayerSerialization )
{
	serializeToFile<CCrfLossLayer>( "FmlCnnCrfLossLayer" );
}

#endif

GTEST_TEST( SerializeFromFile, CrfLossLayerSerialization )
{
	checkSerializeLayer<CCrfLossLayer>( "FmlCnnCrfLossLayer" );
}

// ====================================================================================================================

// CUpsampling2DLayer

template<>
inline void checkSpecificParams<CUpsampling2DLayer>( CUpsampling2DLayer& layer )
{
	ASSERT_EQ( 4, layer.GetHeightCopyCount() );
	ASSERT_EQ( 3, layer.GetWidthCopyCount() );
}

#ifdef GENERATE_SERIALIZATION_FILES

static void setSpecificParams( CUpsampling2DLayer& layer )
{
	layer.SetHeightCopyCount( 4 );
	layer.SetWidthCopyCount( 3 );
}

GTEST_TEST( SerializeToFile, Upsampling2DLayerSerialization )
{
	serializeToFile<CUpsampling2DLayer>( "FmlCnnUpsampling2DLayer" );
}

#endif

GTEST_TEST( SerializeFromFile, Upsampling2DLayerSerialization )
{
	checkSerializeLayer<CUpsampling2DLayer>( "FmlCnnUpsampling2DLayer" );
}

// ====================================================================================================================

// CQualityControlLayer

template<>
inline void checkSpecificParams<CQualityControlLayer>( CQualityControlLayer& layer )
{
	ASSERT_EQ( true, layer.IsResetNeeded() );
}

#ifdef GENERATE_SERIALIZATION_FILES

static void setSpecificParams( CQualityControlLayer& layer )
{
	layer.SetReset( true );
}

GTEST_TEST( SerializeToFile, QualityControlLayerSerialization )
{
	serializeToFile<CAccuracyLayer>( "FmlCnnAccuracyLayer" );
	serializeToFile<CConfusionMatrixLayer>( "FmlCnnConfusionMatrixLayer" );
	serializeToFile<CPrecisionRecallLayer>( "FmlCnnPrecisionRecallLayer" );
}

#endif

GTEST_TEST( SerializeFromFile, QualityControlLayerSerialization )
{
	checkSerializeLayer<CQualityControlLayer>( "FmlCnnAccuracyLayer" );
	checkSerializeLayer<CQualityControlLayer>( "FmlCnnConfusionMatrixLayer" );
	checkSerializeLayer<CQualityControlLayer>( "FmlCnnPrecisionRecallLayer" );
}

// ====================================================================================================================

// CCenterLossLayer

template<>
inline void checkSpecificParams<CCenterLossLayer>( CCenterLossLayer& layer )
{
	ASSERT_EQ( 3, layer.GetNumberOfClasses() );
	ASSERT_NEAR( 0.5, layer.GetClassCentersConvergenceRate(), 1e-3 );
}

#ifdef GENERATE_SERIALIZATION_FILES

static void setSpecificParams( CCenterLossLayer& layer )
{
	layer.SetNumberOfClasses( 3 );
	layer.SetClassCentersConvergenceRate( 0.5 );
}

GTEST_TEST( SerializeToFile, CenterLossLayerSerialization )
{
	serializeToFile<CCenterLossLayer>( "FmlCnnCenterLossLayer" );
}

#endif

GTEST_TEST( SerializeFromFile, CenterLossLayerSerialization )
{
	checkSerializeLayer<CCenterLossLayer>( "FmlCnnCenterLossLayer" );
}

// ====================================================================================================================

// CFocalLossLayer

template<>
inline void checkSpecificParams<CFocalLossLayer>( CFocalLossLayer& layer )
{
	ASSERT_NEAR( 0.5, layer.GetFocalForce(), 1e-3 );
}

template<>
inline void checkSpecificParams<CBinaryFocalLossLayer>( CBinaryFocalLossLayer& layer )
{
	ASSERT_NEAR( 0.5, layer.GetFocalForce(), 1e-3 );
}

#ifdef GENERATE_SERIALIZATION_FILES

static void setSpecificParams( CFocalLossLayer& layer )
{
	layer.SetFocalForce( 0.5 );
}

static void setSpecificParams( CBinaryFocalLossLayer& layer )
{
	layer.SetFocalForce( 0.5 );
}

GTEST_TEST( SerializeToFile, FocalLossLayerSerialization )
{
	serializeToFile<CFocalLossLayer>( "FmlCnnFocalLossLayer" );
	serializeToFile<CBinaryFocalLossLayer>( "FmlCnnBinaryFocalLossLayer" );
}

#endif

GTEST_TEST( SerializeFromFile, FocalLossLayerSerialization )
{
	checkSerializeLayer<CFocalLossLayer>( "FmlCnnFocalLossLayer" );
	checkSerializeLayer<CBinaryFocalLossLayer>( "FmlCnnBinaryFocalLossLayer" );
}

// ====================================================================================================================

// CCtcLossLayer

template<>
inline void checkSpecificParams<CCtcLossLayer>( CCtcLossLayer& layer )
{
	ASSERT_NEAR( 0.5, layer.GetLossWeight(), 1e-3 );
	ASSERT_NEAR( 5., layer.GetMaxGradientValue(), 1e-3 );
	ASSERT_EQ( true, layer.GetAllowBlankLabelSkips() );
	ASSERT_EQ( 2, layer.GetBlankLabel() );
}

#ifdef GENERATE_SERIALIZATION_FILES

static void setSpecificParams( CCtcLossLayer& layer )
{
	layer.SetBlankLabel( 2 );
	layer.SetLossWeight( 0.5 );
	layer.SetAllowBlankLabelSkips( true );
	layer.SetMaxGradientValue( 5. );
}

GTEST_TEST( SerializeToFile, CtcLossLayerSerialization )
{
	serializeToFile<CCtcLossLayer>( "FmlCnnCtcLossLayer" );
}

#endif

GTEST_TEST( SerializeFromFile, CtcLossLayerSerialization )
{
	checkSerializeLayer<CCtcLossLayer>( "FmlCnnCtcLossLayer" );
}

// ====================================================================================================================

// CCtcDecodingLayer

template<>
inline void checkSpecificParams<CCtcDecodingLayer>( CCtcDecodingLayer& layer )
{
	ASSERT_NEAR( 0.5, layer.GetArcProbabilityThreshold(), 1e-3 );
	ASSERT_NEAR( 0.75, layer.GetBlankProbabilityThreshold(), 1e-3 );
	ASSERT_EQ( 2, layer.GetBlankLabel() );
}

#ifdef GENERATE_SERIALIZATION_FILES

static void setSpecificParams( CCtcDecodingLayer& layer )
{
	layer.SetBlankLabel( 2 );
	layer.SetBlankProbabilityThreshold( 0.75 );
	layer.SetArcProbabilityThreshold( 0.5 );
}

GTEST_TEST( SerializeToFile, CtcDecodingLayerSerialization )
{
	serializeToFile<CCtcDecodingLayer>( "FmlCnnCtcDecodingLayer" );
}

#endif

GTEST_TEST( SerializeFromFile, CtcDecodingLayerSerialization )
{
	checkSerializeLayer<CCtcDecodingLayer>( "FmlCnnCtcDecodingLayer" );
}

// ====================================================================================================================

// CPixelToImageLayer

#ifdef GENERATE_SERIALIZATION_FILES

static void setSpecificParams( CPixelToImageLayer& layer )
{
	layer.SetImageHeight( 37 );
	layer.SetImageWidth( 42 );
}

GTEST_TEST( SerializeToFile, PixelToImageLayerSerialization )
{
	serializeToFile<CPixelToImageLayer>( "FmlCnnPixelToImageLayerClass" );
}

#endif

template<>
inline void checkSpecificParams<CPixelToImageLayer>( CPixelToImageLayer& layer )
{
	ASSERT_EQ( 37, layer.GetImageHeight() );
	ASSERT_EQ( 42, layer.GetImageWidth() );
}

GTEST_TEST( SerializeFromFile, PixelToImageLayerSerialization )
{
	checkSerializeLayer<CPixelToImageLayer>( "FmlCnnPixelToImageLayerClass" );
}

// ====================================================================================================================

// CTransposeLayer

#ifdef GENERATE_SERIALIZATION_FILES

static void setSpecificParams( CTransposeLayer& layer )
{
	layer.SetTransposedDimensions( BD_Depth, BD_ListSize );
}

GTEST_TEST( SerializeToFile, TransposeLayerSerialization )
{
	serializeToFile<CTransposeLayer>( "FmlCnnTransposeLayer" );
}

#endif

template<>
inline void checkSpecificParams<CTransposeLayer>( CTransposeLayer& layer )
{
	TBlobDim first = BD_BatchLength;
	TBlobDim second = BD_BatchLength;
	layer.GetTransposedDimensions( first, second );
	ASSERT_EQ( BD_Depth, first );
	ASSERT_EQ( BD_ListSize, second );
}

GTEST_TEST( SerializeFromFile, TransposeLayerSerialization )
{
	checkSerializeLayer<CTransposeLayer>( "FmlCnnTransposeLayer" );
}

// ====================================================================================================================

// CTransformLayer

#ifdef GENERATE_SERIALIZATION_FILES

static void setSpecificParams( CTransformLayer& layer )
{
	layer.SetDimensionRule( BD_BatchLength, CTransformLayer::CDimensionRule( CTransformLayer::O_SetSize, 2 ) );
	layer.SetDimensionRule( BD_BatchWidth, CTransformLayer::CDimensionRule( CTransformLayer::O_Remainder, 1 ) );
	layer.SetDimensionRule( BD_ListSize, CTransformLayer::CDimensionRule( CTransformLayer::O_Multiply, 3 ) );
	layer.SetDimensionRule( BD_Height, CTransformLayer::CDimensionRule( CTransformLayer::O_Divide, 7 ) );
	layer.SetDimensionRule( BD_Width, CTransformLayer::CDimensionRule( CTransformLayer::O_SetSize, 5 ) );
	layer.SetDimensionRule( BD_Channels, CTransformLayer::CDimensionRule( CTransformLayer::O_Divide, 4 ) );
}

GTEST_TEST( SerializeToFile, TransformLayerSerialization )
{
	serializeToFile<CTransformLayer>( "FmlCnnTransformWithoutTransposeLayer" );
}

#endif

template<>
inline void checkSpecificParams<CTransformLayer>( CTransformLayer& layer )
{
	ASSERT_EQ( CTransformLayer::O_SetSize, layer.GetDimensionRule( BD_BatchLength ).Operation );
	ASSERT_EQ( 2, layer.GetDimensionRule( BD_BatchLength ).Parameter );

	ASSERT_EQ( CTransformLayer::O_Remainder, layer.GetDimensionRule( BD_BatchWidth ).Operation );

	ASSERT_EQ( CTransformLayer::O_Multiply, layer.GetDimensionRule( BD_ListSize ).Operation );
	ASSERT_EQ( 3, layer.GetDimensionRule( BD_ListSize ).Parameter );

	ASSERT_EQ( CTransformLayer::O_Divide, layer.GetDimensionRule( BD_Height ).Operation );
	ASSERT_EQ( 7, layer.GetDimensionRule( BD_Height ).Parameter );

	ASSERT_EQ( CTransformLayer::O_SetSize, layer.GetDimensionRule( BD_Width ).Operation );
	ASSERT_EQ( 5, layer.GetDimensionRule( BD_Width ).Parameter );

	ASSERT_EQ( CTransformLayer::O_Divide, layer.GetDimensionRule( BD_Channels ).Operation );
	ASSERT_EQ( 4, layer.GetDimensionRule( BD_Channels ).Parameter );

	// Check default operation
	ASSERT_EQ( CTransformLayer::O_Multiply, layer.GetDimensionRule( BD_Depth ).Operation );
	ASSERT_EQ( 1, layer.GetDimensionRule( BD_Depth ).Parameter );
}

GTEST_TEST( SerializeFromFile, TransformLayerSerialization )
{
	checkSerializeLayer<CTransformLayer>( "FmlCnnTransformWithoutTransposeLayer" );
}

// ====================================================================================================================

// CArgmaxLayer

#ifdef GENERATE_SERIALIZATION_FILES

static void setSpecificParams( CArgmaxLayer& layer )
{
	layer.SetDimension( BD_ListSize );
}

GTEST_TEST( SerializeToFile, ArgmaxLayerSerialization )
{
	serializeToFile<CArgmaxLayer>( "FmlCnnArgmaxLayer" );
}

#endif

template<>
inline void checkSpecificParams<CArgmaxLayer>( CArgmaxLayer& layer )
{
	ASSERT_EQ( BD_ListSize, layer.GetDimension() );
}

GTEST_TEST( SerializeFromFile, ArgmaxLayerSerialization )
{
	checkSerializeLayer<CArgmaxLayer>( "FmlCnnArgmaxLayer" );
}

// ====================================================================================================================

// CAttentionDecoderLayer

#ifdef GENERATE_SERIALIZATION_FILES

static void setSpecificParams( CAttentionDecoderLayer& layer )
{
	layer.SetAttentionScore( AS_DotProduct );
	layer.SetOutputObjectSize( 5 * TestIntValue );
	layer.SetOutputSequenceLen( 7 * TestIntValue );
	layer.SetHiddenLayerSize( 42 * TestIntValue );
}

GTEST_TEST( SerializeToFile, AttentionDecoderLayerSerialization )
{
	serializeToFile<CAttentionDecoderLayer>( "FmlCnnAttentionDecoderLayer" );
}

#endif

template<>
inline void checkSpecificParams<CAttentionDecoderLayer>( CAttentionDecoderLayer& layer )
{
	ASSERT_EQ( 5 * TestIntValue, layer.GetOutputObjectSize() );
	ASSERT_EQ( 7 * TestIntValue, layer.GetOutputSequenceLen() );
	ASSERT_EQ( 42 * TestIntValue, layer.GetHiddenLayerSize() );
}

GTEST_TEST( SerializeFromFile, AttentionDecoderLayerSerialization )
{
	checkSerializeLayer<CAttentionDecoderLayer>( "FmlCnnAttentionDecoderLayer" );
}

// ====================================================================================================================

// CAttentionRecurrentLayer

#ifdef GENERATE_SERIALIZATION_FILES

static void setSpecificParams( CAttentionRecurrentLayer& layer )
{
	layer.SetAttentionScore( AS_DotProduct );
	layer.SetOutputObjectSize( 5 * TestIntValue );
}

GTEST_TEST( SerializeToFile, AttentionRecurrentLayerSerialization )
{
	serializeToFile<CAttentionRecurrentLayer>( "FmlCnnAttentionRecurrentLayer" );
}

#endif

template<>
inline void checkSpecificParams<CAttentionRecurrentLayer>( CAttentionRecurrentLayer& layer )
{
	ASSERT_EQ( 5 * TestIntValue, layer.GetOutputObjectSize() );
}

GTEST_TEST( SerializeFromFile, AttentionRecurrentLayerSerialization )
{
	checkSerializeLayer<CAttentionRecurrentLayer>( "FmlCnnAttentionRecurrentLayer" );
}

// ====================================================================================================================

// CAttentionLayer

#ifdef GENERATE_SERIALIZATION_FILES

static void setSpecificParams( CAttentionLayer& layer )
{
	// AS_Additive is the default attention score of this layer
	// But we use it here in order to test CAttentionLayer::Get*Data methods
	// (otherwise those methods are unavailable)
	layer.SetAttentionScore( AS_Additive );

	CPtr<CDnnBlob> weights = generateBlob( 1, 1, 1, 1, 5 );
	layer.SetFcWeightsData( weights );

	CPtr<CDnnBlob> freeTerms = generateBlob( 1, 1, 1, 1, 1 );
	layer.SetFcFreeTermData( freeTerms );
}

GTEST_TEST( SerializeToFile, AttentionLayerSerialization )
{
	serializeToFile<CAttentionLayer>( "FmlCnnAttentionLayer" );
}

#endif // GENERATE_SERIALIZATION_FILES

template<>
inline void checkSpecificParams<CAttentionLayer>( CAttentionLayer& layer )
{
	checkBlob( *layer.GetFcWeightsData(), 5 );
	checkBlob( *layer.GetFcFreeTermData(), 1 );
}

GTEST_TEST( SerializeFromFile, AttentionLayerSerialization )
{
	checkSerializeLayer<CAttentionLayer>( "FmlCnnAttentionLayer" );
}

// ====================================================================================================================

// CRepeatSequenceLayer

#ifdef GENERATE_SERIALIZATION_FILES

static void setSpecificParams( CRepeatSequenceLayer& layer )
{
	layer.SetRepeatCount( 3 * TestIntValue );
}

GTEST_TEST( SerializeToFile, RepeatSequenceLayerSerialization )
{
	serializeToFile<CRepeatSequenceLayer>( "FmlCnnRepeatSequenceLayer" );
}

#endif // GENERATE_SERIALIZATION_FILES

template<>
inline void checkSpecificParams<CRepeatSequenceLayer>( CRepeatSequenceLayer& layer )
{
	ASSERT_EQ( 3 * TestIntValue, layer.GetRepeatCount() );
}

GTEST_TEST( SerializeFromFile, RepeatSequenceLayerSerialization )
{
	checkSerializeLayer<CRepeatSequenceLayer>( "FmlCnnRepeatSequenceLayer" );
}

// ====================================================================================================================

// CReorgLayer

#ifdef GENERATE_SERIALIZATION_FILES

static void setSpecificParams( CReorgLayer& layer )
{
	layer.SetStride( 3 * TestIntValue );
}

GTEST_TEST( SerializeToFile, ReorgLayerSerialization )
{
	serializeToFile<CReorgLayer>( "FmlCnnCompositeSourceLayer" );
}

#endif // GENERATE_SERIALIZATION_FILES

template<>
inline void checkSpecificParams<CReorgLayer>( CReorgLayer& layer )
{
	ASSERT_EQ( 3 * TestIntValue, layer.GetStride() );
}

GTEST_TEST( SerializeFromFile, ReorgLayerSerialization )
{
	checkSerializeLayer<CReorgLayer>( "FmlCnnReorgLayerClass" );
}

// ====================================================================================================================

// CObjectNormalizationLayer

#ifdef GENERATE_SERIALIZATION_FILES

static void setSpecificParams( CObjectNormalizationLayer& layer )
{
	auto blob = generateBlob( 1, 1, 1, 1, TestSize );
	layer.SetScale( blob );
	layer.SetBias( blob );
	layer.SetEpsilon( 4e-3 );
}

GTEST_TEST( SerializeToFile, ObjectNormalizationLayerSerialization )
{
	serializeToFile<CObjectNormalizationLayer>( "NeoMLDnnObjectNormalizationLayer" );
}

#endif // GENERATE_SERIALIZATION_FILES

template<>
inline void checkSpecificParams<CObjectNormalizationLayer>( CObjectNormalizationLayer& layer )
{
	auto blob = layer.GetScale();
	checkBlob( *blob, TestSize );
	blob = layer.GetBias();
	checkBlob( *blob, TestSize );
	ASSERT_NEAR( layer.GetEpsilon(), 4e-3, 1e-5 );
}

GTEST_TEST( SerializeFromFile, ObjectNormalizationLayerSerialization )
{
	checkSerializeLayer<CObjectNormalizationLayer>( "NeoMLDnnObjectNormalizationLayer" );
}

// ====================================================================================================================

// CMultiheadAttentionLayer

#ifdef GENERATE_SERIALIZATION_FILES

static void setSpecificParams( CMultiheadAttentionLayer& layer )
{
	layer.SetHeadCount( 5 );
	layer.SetHiddenSize( 25 );
	layer.SetDropoutRate( 0.123f );
	layer.SetUseMask( true );
	layer.SetOutputSize( 123 );
}

GTEST_TEST( SerializeToFile, MultiheadAttentionLayerSerialization )
{
	serializeToFile<CMultiheadAttentionLayer>( "NeoMLDnnMultiheadAttentionLayer" );
}

#endif // GENERATE_SERIALIZATION_FILES

template<>
inline void checkSpecificParams<CMultiheadAttentionLayer>( CMultiheadAttentionLayer& layer )
{
	EXPECT_EQ( 5, layer.GetHeadCount() );
	EXPECT_EQ( 25, layer.GetHiddenSize() );
	EXPECT_NEAR( 0.123f, layer.GetDropoutRate(), 1e-5f );
	EXPECT_EQ( true, layer.GetUseMask() );
	EXPECT_EQ( 123, layer.GetOutputSize() );
}

GTEST_TEST( SerializeFromFile, MultiheadAttentionLayerSerialization )
{
	checkSerializeLayer<CMultiheadAttentionLayer>( "NeoMLDnnMultiheadAttentionLayer" );
}

// ====================================================================================================================

// CPositionalEmbeddingLayer

#ifdef GENERATE_SERIALIZATION_FILES

static void setSpecificParams( CPositionalEmbeddingLayer& layer )
{
	layer.SetType( CPositionalEmbeddingLayer::PET_Transformers );
}

GTEST_TEST( SerializeToFile, PositionalEmbeddingLayerSerialization )
{
	serializeToFile<CPositionalEmbeddingLayer>( "NeoMLDnnPositionalEmbeddingLayer" );
}

#endif // GENERATE_SERIALIZATION_FILES

template<>
inline void checkSpecificParams<CPositionalEmbeddingLayer>( CPositionalEmbeddingLayer& layer )
{
	EXPECT_EQ( CPositionalEmbeddingLayer::PET_Transformers, layer.GetType() );
}

GTEST_TEST( SerializeFromFile, PositionalEmbeddingLayerSerialization )
{
	checkSerializeLayer<CPositionalEmbeddingLayer>( "NeoMLDnnPositionalEmbeddingLayer" );
}

// ====================================================================================================================

// CProjectionPoolingLayer

#ifdef GENERATE_SERIALIZATION_FILES

static void setSpecificParams( CProjectionPoolingLayer& layer )
{
	layer.SetDimension( BD_ListSize );
	layer.SetRestoreOriginalImageSize( true );
}

GTEST_TEST( SerializeToFile, ProjectionPoolingLayerSerialization )
{
	serializeToFile<CProjectionPoolingLayer>( "FmlCnnProjectionPoolingLayerClass" );
}

#endif // GENERATE_SERIALIZATION_FILES

template<>
inline void checkSpecificParams<CProjectionPoolingLayer>( CProjectionPoolingLayer& layer )
{
	EXPECT_EQ( BD_ListSize, layer.GetDimension() );
	EXPECT_TRUE( layer.GetRestoreOriginalImageSize() );
}

GTEST_TEST( SerializeFromFile, ProjectionPoolingLayerSerialization )
{
	checkSerializeLayer<CProjectionPoolingLayer>( "FmlCnnProjectionPoolingLayerClass" );
}

// ====================================================================================================================

// CQrnnLayer

#ifdef GENERATE_SERIALIZATION_FILES

static void setSpecificParams( CQrnnLayer& layer )
{
	layer.SetActivation( AF_HardSigmoid );
	layer.SetDropout( 0.05f );
	layer.SetHiddenSize( 15 );
	layer.SetPaddingFront( 4 );
	layer.SetPaddingBack( 3 );
	layer.SetRecurrentMode( CQrnnLayer::RM_BidirectionalConcat );
	layer.SetStride( 2 );
	layer.SetWindowSize( 6 );
}

GTEST_TEST( SerializeToFile, QrnnLayerSerialization )
{
	serializeToFile<CQrnnLayer>( "NeoMLDnnQrnnLayer" );
}

#endif // GENERATE_SERIALIZATION_FILES

template<>
inline void checkSpecificParams<CQrnnLayer>( CQrnnLayer& layer )
{
	EXPECT_EQ( AF_HardSigmoid, layer.GetActivation() );
	EXPECT_EQ( 0.05f, layer.GetDropout() );
	EXPECT_EQ( 15, layer.GetHiddenSize() );
	EXPECT_EQ( 4, layer.GetPaddingFront() );
	EXPECT_EQ( 3, layer.GetPaddingBack() );
	EXPECT_EQ( CQrnnLayer::RM_BidirectionalConcat, layer.GetRecurrentMode() );
	EXPECT_EQ( 2, layer.GetStride() );
	EXPECT_EQ( 6, layer.GetWindowSize() );
}

GTEST_TEST( SerializeFromFile, QrnnLayerSerialization )
{
	checkSerializeLayer<CQrnnLayer>( "NeoMLDnnQrnnLayer" );
}

// ====================================================================================================================

// CQrnnFPoolingLayer

#ifdef GENERATE_SERIALIZATION_FILES

static void setSpecificParams( CQrnnFPoolingLayer& layer )
{
	layer.SetReverse( true);
}

GTEST_TEST( SerializeToFile, QrnnFPoolingLayerSerialization )
{
	serializeToFile<CQrnnFPoolingLayer>( "NeoMLDnnQrnnFPoolingLayer" );
}

#endif // GENERATE_SERIALIZATION_FILES

template<>
inline void checkSpecificParams<CQrnnFPoolingLayer>( CQrnnFPoolingLayer& layer )
{
	EXPECT_TRUE( layer.IsReverse() );
}

GTEST_TEST( SerializeFromFile, QrnnFPoolingLayerLayerSerialization )
{
	checkSerializeLayer<CQrnnFPoolingLayer>( "NeoMLDnnQrnnFPoolingLayer" );
}

// ====================================================================================================================

// CQrnnIfPoolingLayer

#ifdef GENERATE_SERIALIZATION_FILES

static void setSpecificParams( CQrnnIfPoolingLayer& layer )
{
	layer.SetReverse( true);
}

GTEST_TEST( SerializeToFile, QrnnIfPoolingLayerSerialization )
{
	serializeToFile<CQrnnIfPoolingLayer>( "NeoMLDnnQrnnIfPoolingLayer" );
}

#endif // GENERATE_SERIALIZATION_FILES

template<>
inline void checkSpecificParams<CQrnnIfPoolingLayer>( CQrnnIfPoolingLayer& layer )
{
	EXPECT_TRUE( layer.IsReverse() );
}

GTEST_TEST( SerializeFromFile, QrnnIfPoolingLayerLayerSerialization )
{
	checkSerializeLayer<CQrnnIfPoolingLayer>( "NeoMLDnnQrnnIfPoolingLayer" );
}

// ====================================================================================================================

// CTiedEmbeddingsLayer

#ifdef GENERATE_SERIALIZATION_FILES

static void setSpecificParams( CTiedEmbeddingsLayer& layer )
{
	layer.SetEmbeddingsLayerName( "serialization_test_embeddings" );
	layer.SetChannelIndex( 4 );
}

GTEST_TEST( SerializeToFile, TiedEmbeddingsLayerSerialization )
{
	serializeToFile<CTiedEmbeddingsLayer>( "TiedEmbeddingsLayer" );
}

#endif // GENERATE_SERIALIZATION_FILES

template<>
inline void checkSpecificParams<CTiedEmbeddingsLayer>( CTiedEmbeddingsLayer& layer )
{
	EXPECT_EQ( CString( "serialization_test_embeddings" ), layer.GetEmbeddingsLayerName() );
	EXPECT_EQ( 4, layer.GetChannelIndex() );
}

GTEST_TEST( SerializeFromFile, TiedEmbeddingsLayerSerialization )
{
	checkSerializeLayer<CTiedEmbeddingsLayer>( "TiedEmbeddingsLayer" );
}

// ====================================================================================================================

// CIrnnLayer

#ifdef GENERATE_SERIALIZATION_FILES

static void setSpecificParams( CIrnnLayer& layer )
{
	layer.SetHiddenSize( 123 );
	layer.SetIdentityScale( 1e-3f );
	layer.SetInputWeightStd( 0.5f );
}

GTEST_TEST( SerializeToFile, IrnnLayerSerialization )
{
	serializeToFile<CIrnnLayer>( "NeoMLDnnIrnnLayer" );
}

#endif // GENERATE_SERIALIZATION_FILES

template<>
inline void checkSpecificParams<CIrnnLayer>( CIrnnLayer& layer )
{
	EXPECT_EQ( 123, layer.GetHiddenSize() );
	EXPECT_NEAR( 1e-3f, layer.GetIdentityScale(), 1e-6f );
	EXPECT_NEAR( 0.5f, layer.GetInputWeightStd(), 1e-6f );
}

GTEST_TEST( SerializeFromFile, IrnnLayerSerialization )
{
	checkSerializeLayer<CIrnnLayer>( "NeoMLDnnIrnnLayer" );
}

// ====================================================================================================================

// CIndRnnRecurrentLayer

#ifdef GENERATE_SERIALIZATION_FILES

static void setSpecificParams( CIndRnnRecurrentLayer& layer )
{
	layer.SetReverseSequence( true );
	layer.SetDropoutRate( 0.5f );
}

GTEST_TEST( SerializeToFile, IndRnnRecurrentLayerSerialization )
{
	serializeToFile<CIndRnnRecurrentLayer>( "NeoMLDnnIndRnnRecurrentLayer" );
}

#endif // GENERATE_SERIALIZATION_FILES

template<>
inline void checkSpecificParams<CIndRnnRecurrentLayer>( CIndRnnRecurrentLayer& layer )
{
	EXPECT_EQ( true, layer.IsReverseSequence() );
	EXPECT_NEAR( 0.5f, layer.GetDropoutRate(), 1e-6f );
	EXPECT_EQ( AF_Sigmoid, layer.GetActivation() );
}

GTEST_TEST( SerializeFromFile, IndRnnRecurrentLayerSerialization )
{
	checkSerializeLayer<CIndRnnRecurrentLayer>( "NeoMLDnnIndRnnRecurrentLayer" );
}

// ====================================================================================================================

// CIndRnnLayer

#ifdef GENERATE_SERIALIZATION_FILES

static void setSpecificParams( CIndRnnLayer& layer )
{
	layer.SetHiddenSize( 3 );
	layer.SetReverseSequence( true );
	layer.SetDropoutRate( 0.25f );
}

GTEST_TEST( SerializeToFile, IndRnnLayerSerialization )
{
	serializeToFile<CIndRnnLayer>( "NeoMLDnnIndRnnLayer" );
}

#endif // GENERATE_SERIALIZATION_FILES

template<>
inline void checkSpecificParams<CIndRnnLayer>( CIndRnnLayer& layer )
{
	EXPECT_EQ( 3, layer.GetHiddenSize() );
	EXPECT_EQ( true, layer.IsReverseSequence() );
	EXPECT_NEAR( 0.25f, layer.GetDropoutRate(), 1e-6f );
	EXPECT_EQ( AF_Sigmoid, layer.GetActivation() );
}

GTEST_TEST( SerializeFromFile, IndRnnLayerSerialization )
{
	checkSerializeLayer<CIndRnnLayer>( "NeoMLDnnIndRnnLayer" );
}

// ====================================================================================================================

// CDepthToSpaceLayer

#ifdef GENERATE_SERIALIZATION_FILES

static void setSpecificParams( CDepthToSpaceLayer& layer )
{
	layer.SetBlockSize( 3 );
}

GTEST_TEST( SerializeToFile, DepthToSpaceLayerSerialization )
{
	serializeToFile<CDepthToSpaceLayer>( "NeoMLDnnDepthToSpaceLayer" );
}

#endif // GENERATE_SERIALIZATION_FILES

template<>
inline void checkSpecificParams<CDepthToSpaceLayer>( CDepthToSpaceLayer& layer )
{
	EXPECT_EQ( 3, layer.GetBlockSize() );
}

GTEST_TEST( SerializeFromFile, DepthToSpaceLayerSerialization )
{
	checkSerializeLayer<CDepthToSpaceLayer>( "NeoMLDnnDepthToSpaceLayer" );
}

// ====================================================================================================================

// CSpaceToDepthLayer

#ifdef GENERATE_SERIALIZATION_FILES

static void setSpecificParams( CSpaceToDepthLayer& layer )
{
	layer.SetBlockSize( 4 );
}

GTEST_TEST( SerializeToFile, SpaceToDepthLayerSerialization )
{
	serializeToFile<CSpaceToDepthLayer>( "NeoMLDnnSpaceToDepthLayer" );
}

#endif // GENERATE_SERIALIZATION_FILES

template<>
inline void checkSpecificParams<CSpaceToDepthLayer>( CSpaceToDepthLayer& layer )
{
	EXPECT_EQ( 4, layer.GetBlockSize() );
}

GTEST_TEST( SerializeFromFile, SpaceToDepthLayerSerialization )
{
	checkSerializeLayer<CSpaceToDepthLayer>( "NeoMLDnnSpaceToDepthLayer" );
}

// ====================================================================================================================

// CLrnLayer

#ifdef GENERATE_SERIALIZATION_FILES

static void setSpecificParams( CLrnLayer& layer )
{
	layer.SetWindowSize( 4 );
	layer.SetBias( -1.f );
	layer.SetAlpha( 0.4f );
	layer.SetBeta( 0.25f );
}

GTEST_TEST( SerializeToFile, LrnLayerSerialization )
{
	serializeToFile<CLrnLayer>( "NeoMLDnnLrnLayer" );
}

#endif // GENERATE_SERIALIZATION_FILES

template<>
inline void checkSpecificParams<CLrnLayer>( CLrnLayer& layer )
{
	EXPECT_EQ( 4, layer.GetWindowSize() );
	EXPECT_NEAR( -1.f, layer.GetBias(), 1e-6f );
	EXPECT_NEAR( 0.4f, layer.GetAlpha(), 1e-6f );
	EXPECT_NEAR( 0.25f, layer.GetBeta(), 1e-6f );
}

GTEST_TEST( SerializeFromFile, LrnLayerSerialization )
{
	checkSerializeLayer<CLrnLayer>( "NeoMLDnnLrnLayer" );
}

// ====================================================================================================================

// CCastLayer

#ifdef GENERATE_SERIALIZATION_FILES

static void setSpecificParams( CCastLayer& layer )
{
	layer.SetOutputType( CT_Int );
}

GTEST_TEST( SerializeToFile, CastLayerSerialization )
{
	serializeToFile<CCastLayer>( "NeoMLDnnCastLayer" );
}

#endif // GENERATE_SERIALIZATION_FILES

template<>
inline void checkSpecificParams<CCastLayer>( CCastLayer& layer )
{
	EXPECT_EQ( CT_Int, layer.GetOutputType() );
}

GTEST_TEST( SerializeFromFile, CastLayerSerialization )
{
	checkSerializeLayer<CCastLayer>( "NeoMLDnnCastLayer" );
}

// ====================================================================================================================

// CTransformerEncoderLayer

#ifdef GENERATE_SERIALIZATION_FILES

static void setSpecificParams( CTransformerEncoderLayer& layer )
{
	layer.SetHeadCount( 6 );
	layer.SetHiddenSize( 36 );
	layer.SetDropoutRate( 0.2f );
	layer.SetFeedForwardSize( 16 );
}

GTEST_TEST( SerializeToFile, TransformerEncoderLayerSerialization )
{
	serializeToFile<CTransformerEncoderLayer>( "NeoMLDnnTransformerEncoderLayer" );
}

#endif

template<>
inline void checkSpecificParams<CTransformerEncoderLayer>( CTransformerEncoderLayer& layer )
{
	EXPECT_EQ( 6, layer.GetHeadCount() );
	EXPECT_EQ( 36, layer.GetHiddenSize() );
	EXPECT_NEAR( 0.2f, layer.GetDropoutRate(), 1e-6f );
	EXPECT_EQ( 16, layer.GetFeedForwardSize() );
}

GTEST_TEST( SerializeFromFile, TransformerEncoderLayerSerialization )
{
	checkSerializeLayer<CTransformerEncoderLayer>( "NeoMLDnnTransformerEncoderLayer" );
}

// ====================================================================================================================

// CInterpolationLayer

#ifdef GENERATE_SERIALIZATION_FILES

static void setSpecificParams( CInterpolationLayer& layer )
{
	layer.SetRule( BD_ListSize, CInterpolationLayer::CRule::Resize( 71 ) );
	layer.SetRule( BD_Depth, CInterpolationLayer::CRule::Scale( 0.5f ) );
	layer.SetCoords( TInterpolationCoords::AlignCorners );
	layer.SetRound( TInterpolationRound::RoundPreferCeil );
}

GTEST_TEST( SerializeToFile, InterpolationLayerSerialization )
{
	serializeToFile<CInterpolationLayer>( "NeoMLDnnInterpolationLayer" );
}

#endif

template<>
inline void checkSpecificParams<CInterpolationLayer>( CInterpolationLayer& layer )
{
	EXPECT_EQ( CInterpolationLayer::TRuleType::Resize, layer.GetRule( BD_ListSize ).Type );
	EXPECT_EQ( 71, layer.GetRule( BD_ListSize ).NewSize );
	EXPECT_EQ( CInterpolationLayer::TRuleType::Scale, layer.GetRule( BD_Depth ).Type );
	EXPECT_FLOAT_EQ( 0.5f, layer.GetRule( BD_Depth ).ScaleCoeff );
	EXPECT_EQ( TInterpolationCoords::AlignCorners, layer.GetCoords() );
	EXPECT_EQ( TInterpolationRound::RoundPreferCeil, layer.GetRound() );
}

GTEST_TEST( SerializeFromFile, InterpolationLayerSerialization )
{
	checkSerializeLayer<CInterpolationLayer>( "NeoMLDnnInterpolationLayer" );
}

// ====================================================================================================================

// CCumSumLayer

#ifdef GENERATE_SERIALIZATION_FILES

static void setSpecificParams( CCumSumLayer& layer )
{
	layer.SetDimension( BD_ListSize );
	layer.SetReverse( true );
}

GTEST_TEST( SerializeToFile, CumSumLayerSerialization )
{
	serializeToFile<CCumSumLayer>( "NeoMLDnnCumSumLayer" );
}

#endif

template<>
inline void checkSpecificParams<CCumSumLayer>( CCumSumLayer& layer )
{
	EXPECT_EQ( BD_ListSize, layer.GetDimension() );
	EXPECT_TRUE( layer.IsReverse() );
}

GTEST_TEST( SerializeFromFile, CumSumLayerSerialization )
{
	checkSerializeLayer<CCumSumLayer>( "NeoMLDnnCumSumLayer" );
}
