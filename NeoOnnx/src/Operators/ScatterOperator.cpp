/* Copyright © 2017-2022 ABBYY Production LLC

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

#include "../common.h"
#pragma hdrstop

#include "ScatterOperator.h"
#include "NeoOnnxCheck.h"

#include "onnx.pb.h"

namespace NeoOnnx {

class CScatterNDStubLayer : public CBaseLayer {
public:
	explicit CScatterNDStubLayer( IMathEngine& mathEngine ) :
		CBaseLayer( mathEngine, "CScatterNDStubLayer", false ), r( -1 ), q( -1 ) {}

	void Serialize( CArchive& ) override { NeoAssert( false ); }
	void SetR( int newR ) { r = newR; }
	void SetQ( int newQ ) { q = newQ; }

protected:
	void Reshape()
	{
		NeoAssert( r > 0 && q > 0 );
		NeoAssert( inputDescs.Size() == 3 );
		NeoAssert( outputDescs.Size() == 1 );
		NeoAssert( inputDescs[0].GetDataType() == inputDescs[2].GetDataType() );
		NeoAssert( inputDescs[1].GetDataType() == CT_Int );
		outputDescs[0] = inputDescs[0];
	}

	void RunOnce()
	{
		outputBlobs[0]->CopyFrom( inputBlobs[0].Ptr() );
		CFastArray<int, 8> offsets;
		const CBlobDesc& dataDesc = inputBlobs[0]->GetDesc();
		const CBlobDesc& indicesDesc = inputBlobs[1]->GetDesc();
		const CBlobDesc& updatesDesc = inputBlobs[2]->GetDesc();

		const int coordDims = indicesDesc.DimSize( q - 1 );
		offsets.SetSize( coordDims );
		offsets.Last() = 1;
		for( int i = coordDims - 2; i >= 0; --i ) {
			offsets[i] = offsets[i + 1] * dataDesc.DimSize( i + 1 );
		}

		const int updateCount = indicesDesc.BlobSize() / coordDims;
		const int updateSize = updatesDesc.BlobSize() / updateCount;
		const int objectCount = dataDesc.BlobSize() / updateSize;
		NeoAssert( offsets[0] * dataDesc.DimSize( 0 ) == objectCount );

		{
			// DEBUG ONLY
			int debugUpdateSize = 1;
			for( int i = coordDims; i < r; ++i ) {
				debugUpdateSize *= dataDesc.DimSize( i );
			}
			NeoAssert( debugUpdateSize == updateSize );
		}

		CDnnBlobBuffer<int> indices( *inputBlobs[1], 0, inputBlobs[1]->GetDataSize(), TDnnBlobBufferAccess::Read );
		int flatIndex = 0;
		for( int updateIndex = 0; updateIndex < updateCount; ++updateIndex ) {
			int objectIndex = 0;
			for( int coordIndex = 0; coordIndex < coordDims; ++coordIndex ) {
				objectIndex += indices[flatIndex++] * offsets[coordIndex];
			}
			if( outputBlobs[0]->GetDataType() == CT_Float ) {
				MathEngine().VectorCopy( outputBlobs[0]->GetData() + objectIndex * updateSize,
					inputBlobs[2]->GetData() + updateIndex * updateSize, updateSize );
			} else {
				MathEngine().VectorCopy( outputBlobs[0]->GetData<int>() + objectIndex * updateSize,
					inputBlobs[2]->GetData<int>() + updateIndex * updateSize, updateSize );
			}
		}
	}

	void BackwardOnce() override { NeoAssert( false ); }

private:
	int r;
	int q;
};

CScatterNDOperator::CScatterNDOperator( const onnx::NodeProto& scatterND, int opsetVersion ) :
	CLayerOperator( scatterND, opsetVersion )
{
	// v11 - original
	// v13 - support bfloat16
	// v16 - new reduction attribute
	CheckNeoOnnxSupport( OpsetVersion >= 11 && OpsetVersion <= MaxOpsetVersion, "opset version", *this );

	CheckOnnxProtocol( InputCount() == 3, "operator must have 3 inputs", *this );
	CheckOnnxProtocol( OutputCount() == 1, "operator must have 1 output", *this );

	if( OpsetVersion >= 16 ) {
		CString reduction = "none";
		GetAttribute( "reduction", reduction );
		CheckNeoOnnxSupport( reduction == "none", "non-default reduction", *this );
	}
}

void CScatterNDOperator::AddLayers( const CTensorArray& inputs, CDnn& dnn, CTensorArray& outputs ) const
{
	CheckNeoOnnxSupport( inputs[0] != nullptr && inputs[1] != nullptr && inputs[2] != nullptr,
		"optional inputs", *this );
	// DEBUG, delete after figuring out...
	::printf( "%s\n", static_cast<const char*>( Name() ) );
	for( int inputIndex = 0; inputIndex < inputs.Size(); ++inputIndex ) {
		::printf( "\tinput[%d]\t", inputIndex );
		::printf( "( %d", inputs[inputIndex]->Shape()[0] );
		for( int i = 1; i < inputs[inputIndex]->DimCount(); ++i ) {
			::printf( ", %d", inputs[inputIndex]->Shape()[i] );
		}
		::printf( " )\t" );
		if( inputs[inputIndex]->IsCalculated() ) {
			const CDataTensor* dataTensor = dynamic_cast<const CDataTensor*>( inputs[inputIndex].Ptr() );
			::printf( "Blob (%s)\n", dataTensor->Data()->GetDataType() == CT_Float ? "float" : "int" );
		} else {
			::printf( "USER DATA\n" );
		}
	}
	CPtr<const CUserTensor> dataTensor = AsUserTensor( *ConvertTensor( *inputs[0], CTensorLayout::IOLayout( inputs[0]->DimCount() ) ),
		Name() + "_Data", dnn );
	CPtr<const CUserTensor> indicesTensor = AsUserTensor( *ConvertTensor( *inputs[1], CTensorLayout::IOLayout( inputs[1]->DimCount() ) ),
		Name() + "_Indices", dnn );
	CPtr<const CUserTensor> updatesTensor = AsUserTensor( *ConvertTensor( *inputs[2], CTensorLayout::IOLayout( inputs[2]->DimCount() ) ),
		Name() + "_Indices", dnn );

	CPtr<CScatterNDStubLayer> scatterNDStub = new CScatterNDStubLayer( dnn.GetMathEngine() );
	scatterNDStub->SetName( Name() );
	scatterNDStub->SetR( dataTensor->DimCount() );
	scatterNDStub->SetQ( indicesTensor->DimCount() );
	scatterNDStub->Connect( 0, *dataTensor->Layer(), dataTensor->OutputIndex() );
	scatterNDStub->Connect( 1, *indicesTensor->Layer(), indicesTensor->OutputIndex() );
	scatterNDStub->Connect( 2, *updatesTensor->Layer(), indicesTensor->OutputIndex() );
	dnn.AddLayer( *scatterNDStub );
	outputs.Add( new CUserTensor( dataTensor->Shape(), dataTensor->Layout(),
		CLayerOutput( scatterNDStub.Ptr(), 0 ) ) );
}

} // namespace NeoOnnx
