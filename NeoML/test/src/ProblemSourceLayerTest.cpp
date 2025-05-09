/* Copyright © 2024 ABBYY

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

namespace NeoMLTest {

template<class T>
static void recreateLayer( IMathEngine& mathEngine, CPtr<T>& layer,
	const char* name, int batchSize, TBlobType type, CMemoryProblem* problem, CDnn& dnn )
{
	layer->SetName( name );
	layer->SetBatchSize( batchSize );

	CMemoryFile file;
	CArchive archive( &file, CArchive::SD_Storing );
	layer->Serialize( archive );
	archive.Close();
	layer = new T( mathEngine );
	file.SeekToBegin();
	archive.Open( &file, CArchive::SD_Loading );
	layer->Serialize( archive );

	EXPECT_EQ( type, layer->GetLabelType() );
	layer->SetProblem( problem );
	dnn.AddLayer( *layer );
}

// Check for float labels == enumBinarization(int labels).
// As a type T could be CFullyConnectedSourceLayer or CProblemSourceLayer.
template<class T>
static void testLabelTypes( CDnn& dnn, CPtr<T> intLayer, CPtr<T> floatLayer )
{
	const int featureCount = 5;
	const int vectorCount = 10;
	const int classCount = 3;
	const int runCount = 10;
	const int batchSize = ( vectorCount / 2 ) - 1;

	static_assert( vectorCount >= classCount, "" );
	static_assert( classCount > 2, "" );

	CPtr<CMemoryProblem> problem = new CMemoryProblem( featureCount, classCount );
	for( int i = 0; i < vectorCount; ++i ) {
		CSparseFloatVector vector;
		const int index = dnn.Random().UniformInt( 0, featureCount - 1 );
		vector.SetAt( index, 1.f );
		problem->Add( vector, i % classCount );
	}

	intLayer->SetLabelType( CT_Int );
	recreateLayer( dnn.GetMathEngine(), intLayer, "intLayer", batchSize, CT_Int, problem.Ptr(), dnn );
	recreateLayer( dnn.GetMathEngine(), floatLayer, "floatLayer", batchSize, CT_Float, problem.Ptr(), dnn );

	CPtr<CEnumBinarizationLayer> enumBin = EnumBinarization( classCount )( CDnnLayerLink( intLayer.Ptr(), 1 ) );

	Sink( CDnnLayerLink( floatLayer.Ptr(), 0 ), "floatData" );
	CPtr<CSinkLayer> floatLabel = Sink( CDnnLayerLink( floatLayer.Ptr(), 1 ), "floatLabel" );
	Sink( CDnnLayerLink( floatLayer.Ptr(), 2 ), "floatWeights" );

	Sink( CDnnLayerLink( intLayer.Ptr(), 0 ), "intData" );
	CPtr<CSinkLayer> intLabel = Sink( enumBin.Ptr(), "intLabel" );
	Sink( CDnnLayerLink( intLayer.Ptr(), 2 ), "intWeights" );

	for( int run = 0; run < runCount; ++run ) {
		dnn.RunOnce();

		CArray<float> expected;
		expected.SetSize( floatLabel->GetBlob()->GetDataSize() );
		floatLabel->GetBlob()->CopyTo( expected.GetPtr() );

		CArray<float> result;
		result.SetSize( intLabel->GetBlob()->GetDataSize() );
		intLabel->GetBlob()->CopyTo( result.GetPtr() );

		EXPECT_EQ( classCount * batchSize, expected.Size() );
		EXPECT_EQ( expected.Size(), result.Size() );
		for( int i = 0; i < expected.Size(); ++i ) {
			EXPECT_FLOAT_EQ( expected[i], result[i] );
		}
	}
}

} // namespace NeoMLTest

//---------------------------------------------------------------------------------------------------------------------

TEST( CDnnProblemTest, LabelTypes )
{
	CRandom random( 0x0CA );
	{
		CDnn dnn( random, MathEngine() );

		CPtr<CProblemSourceLayer> intSource = new CProblemSourceLayer( MathEngine() );
		CPtr<CProblemSourceLayer> floatSource = new CProblemSourceLayer( MathEngine() );

		testLabelTypes( dnn, intSource, floatSource );
	}
	{
		CDnn dnn( random, MathEngine() );

		CPtr<CFullyConnectedSourceLayer> intFc = new CFullyConnectedSourceLayer( MathEngine() );
		intFc->SetNumberOfElements( 3 );
		CPtr<CFullyConnectedSourceLayer> floatFc = new CFullyConnectedSourceLayer( MathEngine() );
		floatFc->SetNumberOfElements( 3 );

		testLabelTypes( dnn, intFc, floatFc );
	}
}
