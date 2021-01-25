/* Copyright © 2017-2021 ABBYY Production LLC

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

#include "PyDnnBlob.h"

py::array CreateArray( const CDnnBlob& blob )
{
	std::vector<int> shape;

	for( int i = 0; i < 7; i++ ) {
		if( blob.DimSize(i) != 1 ) {
			for( int j = i; j < 7; j++ ) {
				shape.push_back( blob.DimSize(j) );
			}
			break;
		}
	}

	if( shape.empty() ) {
		shape.push_back( 1 );
	}

	if( blob.GetDataType() == CT_Float ) {
		py::array_t<float, py::array::c_style> result( shape );
		auto temp = result.mutable_unchecked();
		blob.CopyTo( (float*)temp.data() );
		return result;
	} else if( blob.GetDataType() == CT_Int ) {
		py::array_t<int, py::array::c_style> result( shape );
		auto temp = result.mutable_unchecked();
		blob.CopyTo( (int*)temp.data() );
		return result;
	}
	assert( false );
	return py::array();
}

CPtr<CDnnBlob> CreateBlob( IMathEngine& mathEngine, const py::array& data )
{
	TBlobType blobType = data.dtype().kind() == 'f' ? CT_Float : CT_Int;

	std::vector<int> shape;
	switch( data.ndim() ) {
		case 1:
			shape = {1, 1, 1, 1, 1, 1, (int)data.shape(0)};
			break;
		case 2:
			shape = {1, 1, 1, 1, 1, (int)data.shape(0), (int)data.shape(1)};
			break;
		case 3:
			shape = {1, 1, 1, 1, (int)data.shape(0), (int)data.shape(1), (int)data.shape(2)};
			break;
		case 4:
			shape = {1, 1, 1, (int)data.shape(0), (int)data.shape(1), (int)data.shape(2), (int)data.shape(3)};
			break;
		case 5:
			shape = {1, 1, (int)data.shape(0), (int)data.shape(1), (int)data.shape(2), (int)data.shape(3), (int)data.shape(4)};
			break;
		case 6:
			shape = {1, (int)data.shape(0), (int)data.shape(1), (int)data.shape(2), (int)data.shape(3), (int)data.shape(4), (int)data.shape(5)};
			break;
		case 7:
			shape = {(int)data.shape(0), (int)data.shape(1), (int)data.shape(2), (int)data.shape(3), (int)data.shape(4), (int)data.shape(5), (int)data.shape(6)};
			break;
		default:
			assert( false );
	};

	CPtr<CDnnBlob> blob = CDnnBlob::CreateTensor( mathEngine, blobType, { shape[0], shape[1], shape[2], shape[3], shape[4], shape[5], shape[6] } );
	if( blobType == CT_Float ) {
		blob->CopyFrom( (float*)data.data() );
	} else if( blobType == CT_Int ) {
		blob->CopyFrom( (int*)data.data() );
	} else {
		assert( false );
	}
	return blob;
}

//------------------------------------------------------------------------------------------------------------

CPyDnnBlob::CPyDnnBlob( const CPyMathEngine& pyMathEngine, TBlobType type, int batchLength, int batchWidth, int listSize,
		int height, int width, int depth, int channels ) :
	mathEngineOwner( &pyMathEngine.MathEngineOwner() ),
	blob( CDnnBlob::CreateTensor( mathEngineOwner->MathEngine(), type, { batchLength, batchWidth, listSize, height, width, depth, channels } ) )
{
}

CPyDnnBlob::CPyDnnBlob( const CPyMathEngine& pyMathEngine, TBlobType blobType, int batchLength, int batchWidth, int listSize,
		int height, int width, int depth, int channels, py::array data ) :
	mathEngineOwner( &pyMathEngine.MathEngineOwner() ),
	blob( CDnnBlob::CreateTensor( mathEngineOwner->MathEngine(), blobType, { batchLength, batchWidth, listSize, height, width, depth, channels } ) )
{
	if( blobType == CT_Float ) {
		blob->CopyFrom( (float*)data.data() );
	} else if( blobType == CT_Int ) {
		blob->CopyFrom( (int*)data.data() );
	} else {
		assert( false );
	}
}

CPyDnnBlob::CPyDnnBlob( const CPyMathEngine& pyMathEngine, py::array data ) :
	mathEngineOwner( &pyMathEngine.MathEngineOwner() ),
	blob( CreateBlob( mathEngineOwner->MathEngine(), data ) )
{
}

CPyDnnBlob::CPyDnnBlob( CPyMathEngineOwner& _mathEngineOwner, CDnnBlob& _blob ) :
	mathEngineOwner( &_mathEngineOwner ),
	blob( &_blob )
{
}

CPyDnnBlob::CPyDnnBlob( const CPyMathEngine& pyMathEngine, const std::string& path ) :
	mathEngineOwner( &pyMathEngine.MathEngineOwner() )
{
}

py::tuple CPyDnnBlob::GetShape() const
{
	auto t = py::tuple(7);
	for( int i = 0; i < t.size(); ++i ) {
		t[i] = DimSize(i);
	}
	return t;
}

//------------------------------------------------------------------------------------------------------------

void InitializeBlob( py::module& m )
{
	py::class_<CPyDnnBlob>(m, "Blob")
		.def( py::init([]( const CPyDnnBlob& blob )
		{
			return new CPyDnnBlob( blob.MathEngineOwner(), blob.Blob() );
		}) )
		.def( py::init([]( const CPyMathEngine& mathEngine, const std::string& path )
		{
			return new CPyDnnBlob( mathEngine, path );
		}) )
		.def( "shape", &CPyDnnBlob::GetShape, py::return_value_policy::reference )
		.def( "batch_len", &CPyDnnBlob::GetBatchLength, py::return_value_policy::reference )
		.def( "batch_width", &CPyDnnBlob::GetBatchWidth, py::return_value_policy::reference )
		.def( "list_size", &CPyDnnBlob::GetListSize, py::return_value_policy::reference )
		.def( "height", &CPyDnnBlob::GetHeight, py::return_value_policy::reference )
		.def( "width", &CPyDnnBlob::GetWidth, py::return_value_policy::reference )
		.def( "depth", &CPyDnnBlob::GetDepth, py::return_value_policy::reference )
		.def( "channels", &CPyDnnBlob::GetChannelsCount, py::return_value_policy::reference )

		.def( "size", &CPyDnnBlob::GetDataSize, py::return_value_policy::reference )
		.def( "object_count", &CPyDnnBlob::GetObjectCount, py::return_value_policy::reference )
		.def( "object_size", &CPyDnnBlob::GetObjectSize, py::return_value_policy::reference )
		.def( "geometrical_size", &CPyDnnBlob::GetGeometricalSize, py::return_value_policy::reference )
		.def( "data", &CPyDnnBlob::GetData, py::return_value_policy::reference )
	;

	m.def("tensor", []( const CPyMathEngine& mathEngine, py::tuple shape, const std::string& blob_type ) {
		TBlobType blobType = CT_Float;
		if( blob_type == "int32" ) {
			blobType = CT_Int;
		}
		return new CPyDnnBlob( mathEngine, blobType, shape[6].cast<int>(), shape[5].cast<int>(), shape[4].cast<int>(),
			shape[3].cast<int>(), shape[2].cast<int>(), shape[1].cast<int>(), shape[0].cast<int>() );
	});

	m.def("tensor", []( const CPyMathEngine& mathEngine, py::tuple shape, const std::string& blob_type, py::array data ) {
		TBlobType blobType = CT_Float;
		if( blob_type == "int32" ) {
			blobType = CT_Int;
		}
		return new CPyDnnBlob( mathEngine, blobType, shape[6].cast<int>(), shape[5].cast<int>(), shape[4].cast<int>(),
			shape[3].cast<int>(), shape[2].cast<int>(), shape[1].cast<int>(), shape[0].cast<int>(), data );
	});

	m.def("tensor", []( const CPyMathEngine& mathEngine, py::array data ) {
		return new CPyDnnBlob( mathEngine, data );
	});
}
