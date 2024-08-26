/* Copyright © 2017-2024 ABBYY

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
#include "PyMemoryFile.h"

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

class CPyMemoryHandle : public CMemoryHandle {
public:
	CPyMemoryHandle( IMathEngine* mathEngine, const void* object ) :
		CMemoryHandle( mathEngine, object, 0 )
	{
	}

	void* GetPtr() const { return const_cast<void*>( CMemoryHandle::Object ); }
};

static CBlobDesc createBlobDesc( TBlobType type, std::initializer_list<int> dimensions )
{
	CBlobDesc desc;
	desc.SetDataType( type );
	for( int i = 0; i < static_cast<int>(dimensions.size()); i++ ) {
		desc.SetDimSize(i, dimensions.begin()[i]);
	}
	return desc;
}

class CPyDnnBlob : public CDnnBlob {
public:
	CPyDnnBlob( IMathEngine& mathEngine, TBlobType type, std::initializer_list<int> dimension, py::buffer_info&& _info );
	virtual ~CPyDnnBlob();

	CPyDnnBlob( const CPyDnnBlob& ) = delete;
	CPyDnnBlob& operator=( const CPyDnnBlob& ) = delete;

private:
	py::buffer_info* info;
};

CPyDnnBlob::CPyDnnBlob( IMathEngine& mathEngine, TBlobType type, std::initializer_list<int> dimension, py::buffer_info&& _info ) :
	CDnnBlob( mathEngine, createBlobDesc( type, dimension ), CPyMemoryHandle( &mathEngine, _info.ptr ), false ),
	info( new py::buffer_info( std::move( _info ) ) )
{
}

CPyDnnBlob::~CPyDnnBlob()
{
	// py::buffer_info holds reference to the corresponding Python object
	// This reference will be released during py::~buffer_info
	// All the reference counting must be done under GIL
	py::gil_scoped_acquire acquire;
	delete info;
}

//------------------------------------------------------------------------------------------------------------

CPyBlob::CPyBlob( const CPyMathEngine& pyMathEngine, TBlobType type, int batchLength, int batchWidth, int listSize,
		int height, int width, int depth, int channels ) :
	mathEngineOwner( &pyMathEngine.MathEngineOwner() ),
	blob( CDnnBlob::CreateTensor( mathEngineOwner->MathEngine(), type, { batchLength, batchWidth, listSize, height, width, depth, channels } ) )
{
}

CPyBlob::CPyBlob( const CPyMathEngine& pyMathEngine, TBlobType blobType, int batchLength, int batchWidth, int listSize,
		int height, int width, int depth, int channels, py::buffer data, bool copy ) :
	mathEngineOwner( &pyMathEngine.MathEngineOwner() ),
	blob()
{
	if( copy ) {
		blob = CDnnBlob::CreateTensor( mathEngineOwner->MathEngine(), blobType, { batchLength, batchWidth, listSize, height, width, depth, channels } );
		py::buffer_info info = data.request();
		if( blobType == CT_Float ) {
			blob->CopyFrom( (float*)info.ptr );
		} else if( blobType == CT_Int ) {
			blob->CopyFrom( (int*)info.ptr );
		} else {
			assert( false );
		}
	} else {
		blob = new CPyDnnBlob( mathEngineOwner->MathEngine(), blobType, { batchLength, batchWidth, listSize, height, width, depth, channels }, data.request() );
	}
}

CPyBlob::CPyBlob( CPyMathEngineOwner& _mathEngineOwner, CDnnBlob* _blob ) :
	mathEngineOwner( &_mathEngineOwner ),
	blob( _blob )
{
}

py::object CPyBlob::GetMathEngine() const
{
	CPyMathEngine mathEngine( *mathEngineOwner );
	py::object m = py::module::import("neoml.MathEngine");
	if( mathEngineOwner->MathEngine().GetType() == MET_Cpu ) {
		py::object constructor = m.attr( "CpuMathEngine" );
		return constructor( mathEngine );
	}
	py::object constructor = m.attr( "GpuMathEngine" );
	return constructor( mathEngine );
}

py::buffer_info CPyBlob::GetBufferInfo() const
{
	if( blob == 0 ) {
		return py::buffer_info();
	}

	void* ptr = nullptr;
	CMemoryHandle data;
	if( blob->GetDataType() == CT_Float ) {
		CFloatHandle floatData = blob->GetData<float>();
		data = *static_cast<CMemoryHandle*>(&floatData);
	} else {
		CIntHandle intData = blob->GetData<int>();
		data = *static_cast<CMemoryHandle*>(&intData);
	}
	ptr = static_cast<CPyMemoryHandle*>( &data )->GetPtr();

	std::vector<size_t> shape;
	for( int i = 0; i < 7; i++ ) {
		if( blob->DimSize(i) != 1 ) {
			shape.push_back( static_cast<size_t>(blob->DimSize(i)) );
		}
	}

	if( shape.empty() ) {
		shape.push_back( 1 );
	}

	const int shapeSize = static_cast<int>( shape.size() );
	std::array<std::array<size_t,7>, 7> multShape;
	for( int i = 0; i < shapeSize; i++ ) {
		multShape[i][i] = shape[i];
		for( int j = i + 1; j < shapeSize; j++ ) {
			multShape[i][j] = multShape[i][j-1] * shape[j];
		}
	}
	size_t itemSize = blob->GetDataType() == CT_Float ? sizeof(float) : sizeof(int);
	std::string format = blob->GetDataType() == CT_Float ? py::format_descriptor<float>::format() : py::format_descriptor<int>::format();

	switch( shapeSize ) {
		case 1:
			return py::buffer_info( ptr, itemSize, format, 1, std::vector<size_t>{shape[0]},
				std::vector<size_t>{itemSize} );
		case 2:
			return py::buffer_info( ptr, itemSize, format, 2, { shape[0], shape[1] },
				{ itemSize * multShape[1][1], itemSize } );
		case 3:
			return py::buffer_info( ptr, itemSize, format, 3, { shape[0], shape[1], shape[2] },
				{ itemSize * multShape[1][2], itemSize * multShape[2][2], itemSize } );
		case 4:
			return py::buffer_info( ptr, itemSize, format, 4, { shape[0], shape[1], shape[2], shape[3] },
				{ itemSize * multShape[1][3], itemSize * multShape[2][3], itemSize * multShape[3][3], itemSize } );
		case 5:
			return py::buffer_info( ptr, itemSize, format, 5, { shape[0], shape[1], shape[2], shape[3], shape[4] },
				{ itemSize * multShape[1][4], itemSize * multShape[2][4], itemSize * multShape[3][4], itemSize * multShape[4][4], itemSize } );
		case 6:
			return py::buffer_info( ptr, itemSize, format, 6, { shape[0], shape[1], shape[2], shape[3], shape[4], shape[5] },
				{ itemSize * multShape[1][5], itemSize * multShape[2][5], itemSize * multShape[3][5], itemSize * multShape[4][5], itemSize * multShape[5][5], itemSize } );
		case 7:
			return py::buffer_info( ptr, itemSize, format, 7, { shape[0], shape[1], shape[2], shape[3], shape[4], shape[5], shape[6] },
				{ itemSize * multShape[1][6], itemSize * multShape[2][6], itemSize * multShape[3][6], itemSize * multShape[4][6], itemSize * multShape[5][6], itemSize * multShape[6][6], itemSize } );
		default:
			assert( false );
	};
	return py::buffer_info();
}

CPyBlob CPyBlob::Copy( const CPyMathEngine& pyMathEngine ) const
{
	if( blob == 0 ) {
		return CPyBlob( pyMathEngine.MathEngineOwner(), 0 );
	}

	CPyBlob res(pyMathEngine, blob->GetDataType(), blob->GetBatchLength(), blob->GetBatchWidth(), blob->GetListSize(),
		blob->GetHeight(), blob->GetWidth(), blob->GetDepth(), blob->GetChannelsCount());
	if( &res.Blob()->GetMathEngine() == &blob->GetMathEngine() ) {
		res.Blob()->CopyFrom( blob );
	} if( blob->GetDataType() == CT_Float ) {
		float* buffer = blob->GetBuffer<float>( 0, blob->GetDataSize(), true );
		res.Blob()->CopyFrom( buffer );
		blob->ReleaseBuffer( buffer, false );
	} else {
		int* buffer = blob->GetBuffer<int>( 0, blob->GetDataSize(), true );
		res.Blob()->CopyFrom( buffer );
		blob->ReleaseBuffer( buffer, false );
	}
	return res;		
}

py::tuple CPyBlob::GetShape() const
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
	py::class_<CPyBlob>(m, "Blob", py::buffer_protocol())
		.def( py::init([]( const CPyBlob& blob )
		{
			return CPyBlob( blob.MathEngineOwner(), blob.Blob() );
		}) )
		.def( "math_engine", &CPyBlob::GetMathEngine, py::return_value_policy::reference )
		.def( "shape", &CPyBlob::GetShape, py::return_value_policy::reference )
		.def( "batch_len", &CPyBlob::GetBatchLength, py::return_value_policy::reference )
		.def( "batch_width", &CPyBlob::GetBatchWidth, py::return_value_policy::reference )
		.def( "list_size", &CPyBlob::GetListSize, py::return_value_policy::reference )
		.def( "height", &CPyBlob::GetHeight, py::return_value_policy::reference )
		.def( "width", &CPyBlob::GetWidth, py::return_value_policy::reference )
		.def( "depth", &CPyBlob::GetDepth, py::return_value_policy::reference )
		.def( "channels", &CPyBlob::GetChannelsCount, py::return_value_policy::reference )

		.def( "size", &CPyBlob::GetDataSize, py::return_value_policy::reference )
		.def( "object_count", &CPyBlob::GetObjectCount, py::return_value_policy::reference )
		.def( "object_size", &CPyBlob::GetObjectSize, py::return_value_policy::reference )

		.def( "copy", &CPyBlob::Copy, py::return_value_policy::reference )

		.def_buffer([](CPyBlob& blob) -> py::buffer_info
		{
			return blob.GetBufferInfo();
		})

		.def(py::pickle(
			[](const CPyBlob& pyBlob) {
				CPyMemoryFile file;
				CArchive archive( &file, CArchive::store );
				bool isNull = (pyBlob.Blob() == 0);
				archive << isNull;
				if( pyBlob.Blob() != 0 ) {
					pyBlob.Blob()->Serialize( archive );
				}
				archive.Close();
				file.Close();
				return py::make_tuple( file.GetBuffer() );
			},
			[](py::tuple t) {
				if( t.size() != 1 ) {
					throw std::runtime_error("Invalid state!");
				}

				auto t0_array = t[0].cast<py::array>();
				CPyMemoryFile file( t0_array );
				CArchive archive( &file, CArchive::load );
				bool isNull = true;
				archive >> isNull;
				CPtr<CPyMathEngineOwner> mathEngineOwner( new CPyMathEngineOwner() );
				CPtr<CDnnBlob> blob;		
				if( !isNull ) {
					blob = new CDnnBlob( mathEngineOwner->MathEngine() );
					blob->Serialize( archive );
				}				
				return CPyBlob( *mathEngineOwner.Ptr(), blob.Ptr() );
			}
		))
	;

	m.def("store_blob", []( const CPyBlob& blob, const std::string& file_path ) {
		CArchiveFile file( file_path.c_str(), CArchive::store );
		CArchive archive( &file, CArchive::store );
		blob.Blob()->Serialize( archive );
	});

	m.def("load_blob", []( const CPyMathEngine& mathEngine, const std::string& file_path ) {
		CArchiveFile file( file_path.c_str(), CArchive::load );
		CArchive archive( &file, CArchive::load );
		CPtr<CDnnBlob> blob( new CDnnBlob(mathEngine.MathEngineOwner().MathEngine()) );
		blob->Serialize( archive );
		return CPyBlob( mathEngine.MathEngineOwner(), blob );
	});

	m.def("tensor", []( const CPyMathEngine& mathEngine, py::array shapes, const std::string& blob_type ) {
		const int* shape = reinterpret_cast<const int*>( shapes.data() );
		TBlobType blobType = CT_Float;
		if( blob_type == "int32" ) {
			blobType = CT_Int;
		}
		return CPyBlob( mathEngine, blobType, shape[0], shape[1], shape[2], shape[3], shape[4], shape[5], shape[6] );
	});

	m.def("tensor", []( const CPyMathEngine& mathEngine, py::array shapes, const std::string& blob_type, py::buffer buffer, bool copy ) {
		const int* shape = reinterpret_cast<const int*>( shapes.data() );
		TBlobType blobType = CT_Float;
		if( blob_type == "int32" ) {
			blobType = CT_Int;
		}
		return CPyBlob( mathEngine, blobType, shape[0], shape[1], shape[2], shape[3], shape[4], shape[5], shape[6], buffer, copy );
	});
}