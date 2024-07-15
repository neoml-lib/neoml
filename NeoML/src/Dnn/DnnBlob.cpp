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

#include <NeoML/Dnn/DnnBlob.h>
#include <NeoMathEngine/NeoMathEngine.h>
#include <NeoML/Dnn/Layers/LossLayer.h>

namespace NeoML {

CDnnBlob::CDnnBlob( IMathEngine& _mathEngine ) :
	mathEngine( _mathEngine ),
	dataOwned( true ),
	parent(0),
	parentPos(0)
{
}

CDnnBlob::CDnnBlob( CDnnBlob&& other ) :
	mathEngine( other.mathEngine ),
	desc( std::move( other.desc ) ),
	data( std::move( other.data ) ),
	dataOwned( other.dataOwned ),
	parent( other.parent ),
	parentPos( other.parentPos )
{
	if( !data.IsNull() && parent == nullptr && dataOwned ) {
		TransferDataToThisThread();
	}
	other.dataOwned = false; // ensure, no premature free
}

CDnnBlob& CDnnBlob::operator=( CDnnBlob&& other )
{
	if( this != &other ) {
		if( !data.IsNull() && parent == nullptr && dataOwned ) {
			mathEngine.HeapFree( data );
		}

		NeoAssert( &mathEngine == &other.mathEngine );
		desc = std::move( other.desc );
		data = std::move( other.data );
		dataOwned = std::move( other.dataOwned );
		parent = std::move( other.parent );
		parentPos = std::move( other.parentPos );

		if( !data.IsNull() && parent == nullptr && dataOwned ) {
			TransferDataToThisThread();
		}
		other.dataOwned = false; // ensure, no premature free
	}
	return *this;
}

CDnnBlob* CDnnBlob::CreateVector(IMathEngine& mathEngine, TBlobType type, int vectorSize)
{
	CDnnBlob* result = FINE_DEBUG_NEW CDnnBlob( mathEngine );
	result->initializeTensor(type, {vectorSize});
	return result;
}

CDnnBlob* CDnnBlob::CreateMatrix(IMathEngine& mathEngine, TBlobType type, int matrixHeight, int matrixWidth)
{
	CDnnBlob* result = FINE_DEBUG_NEW CDnnBlob( mathEngine );
	result->initializeTensor(type, {matrixHeight, matrixWidth});
	return result;
}

CDnnBlob* CDnnBlob::CreateTensor(IMathEngine& mathEngine, TBlobType type, std::initializer_list<int> dimensions)
{
	CDnnBlob* result = FINE_DEBUG_NEW CDnnBlob( mathEngine );
	result->initializeTensor(type, dimensions);
	return result;
}

CDnnBlob* CDnnBlob::CreateDataBlob(IMathEngine& mathEngine, TBlobType type, int batchLength, int batchWidth, int channelsCount)
{
	CDnnBlob* result = FINE_DEBUG_NEW CDnnBlob( mathEngine );
	result->initializeBlob(type, batchLength, batchWidth, 1, 1, 1, 1, channelsCount);
	return result;
}

CDnnBlob* CDnnBlob::CreateListBlob(IMathEngine& mathEngine, TBlobType type, int batchLength, int batchWidth,
	int listSize, int channelsCount)
{
	CDnnBlob* result = FINE_DEBUG_NEW CDnnBlob( mathEngine );
	result->initializeBlob(type, batchLength, batchWidth, listSize, 1, 1, 1, channelsCount);
	return result;
}

CDnnBlob* CDnnBlob::Create2DImageBlob(IMathEngine& mathEngine, TBlobType type, int batchLength, int batchWidth, 
	int imageHeight, int imageWidth, int channelsCount)
{
	CDnnBlob* result = FINE_DEBUG_NEW CDnnBlob( mathEngine );
	result->initializeBlob( type, batchLength, batchWidth, 1, imageHeight, imageWidth, 1, 
		channelsCount);
	return result;
}

CDnnBlob* CDnnBlob::Create3DImageBlob(IMathEngine& mathEngine, TBlobType type, int batchLength, int batchWidth,
	int imageHeight, int imageWidth, int imageDepth, int channelsCount)
{
	CDnnBlob* result = FINE_DEBUG_NEW CDnnBlob( mathEngine );
	result->initializeBlob( type, batchLength, batchWidth, 1, imageHeight, imageWidth, 
		imageDepth, channelsCount);
	return result;
}

CDnnBlob* CDnnBlob::CreateWindowBlob(const CPtr<CDnnBlob>& parent, int windowSize)
{
	CDnnBlob* result = FINE_DEBUG_NEW CDnnBlob( parent->GetMathEngine() );
	result->initializeWindow(parent, windowSize);
	return result;
}

CDnnBlob* CDnnBlob::CreateBlob(IMathEngine& mathEngine, TBlobType type, const CBlobDesc& pattern)
{
	CDnnBlob* result = FINE_DEBUG_NEW CDnnBlob( mathEngine );
	result->initializeByPattern(type, pattern);
	return result;
}

void CDnnBlob::initializeBlob(TBlobType type,
	int batchLength, int batchWidth, int listSize, int height, int width, int depth, int channels)
{
	CBlobDesc pattern( { batchLength, batchWidth, listSize, height, width, depth, channels } );
	initializeByPattern( type, pattern );
}

void CDnnBlob::initializeTensor(TBlobType type, std::initializer_list<int> dimensions)
{
	NeoAssert(dimensions.size() <= CBlobDesc::MaxDimensions);
	CBlobDesc pattern( type );
	for( int i = 0; i < static_cast<int>( dimensions.size() ); ++i ) {
		pattern.SetDimSize( i, dimensions.begin()[i] );
	}
	initializeByPattern( type, pattern );
}

void CDnnBlob::initializeWindow(const CPtr<CDnnBlob>& _parent, int windowSize)
{
	NeoAssert(desc.GetDataType() == CT_Invalid);

	parent = _parent;
	desc = parent->GetDesc();
	desc.SetDimSize(BD_BatchLength, windowSize);

	// Initializing data pointer
	SetParentPos( 0 );
}

void CDnnBlob::initializeByPattern(TBlobType type, const CBlobDesc& pattern)
{
	NeoAssert(desc.GetDataType() == CT_Invalid);

	const int size = pattern.BlobSize();
	switch(type) {
		case CT_Float:
			data = mathEngine.HeapAllocTyped<float>( size );
			break;
		case CT_Int:
			data = mathEngine.HeapAllocTyped<int>( size );
			break;
		default:
			NeoAssert( false );
	}
	desc = pattern;
	desc.SetDataType( type );
}

CDnnBlob::~CDnnBlob()
{
	if( !data.IsNull() && parent == 0 && dataOwned ) {
		mathEngine.HeapFree( data );
	}
}

void CDnnBlob::ReleaseBuffer( void* ptr, bool exchange )
{
	mathEngine.ReleaseBuffer( data, ptr, exchange );
}

CDnnBlob* CDnnBlob::GetClone(TBlobType _type) const
{
	CDnnBlob* result = FINE_DEBUG_NEW CDnnBlob( mathEngine );
	result->initializeByPattern(_type, desc);
	return result;
}

CDnnBlob* CDnnBlob::GetClone() const
{
	return GetClone(GetDataType());
}

CDnnBlob* CDnnBlob::GetCopy() const
{
	CDnnBlob* copy = GetClone(GetDataType());
	copy->CopyFrom(this);
	return copy;
}

void CDnnBlob::CopyFrom(const CDnnBlob* other)
{
	NeoAssert( other != nullptr );
	NeoAssert( GetDataType() == other->GetDataType() );
	NeoAssert( HasEqualDimensions( other ) );
	if( this == other ) {
		return;
	}
	switch( GetDataType() ) {
		case CT_Float:
			if( &mathEngine == &other->GetMathEngine() ) {
				mathEngine.VectorCopy( GetData<float>(), other->GetData<float>(), GetDataSize() );
			} else {
				CDnnBlobBuffer<float> buffer( const_cast<CDnnBlob&>( *other ), TDnnBlobBufferAccess::Read );
				CopyFrom( buffer.Ptr() );
			}
			break;
		case CT_Int:
			if( &mathEngine == &other->GetMathEngine() ) {
				mathEngine.VectorCopy( GetData<int>(), other->GetData<int>(), GetDataSize() );
			} else {
				CDnnBlobBuffer<int> buffer( const_cast<CDnnBlob&>( *other ), TDnnBlobBufferAccess::Read );
				CopyFrom( buffer.Ptr() );
			}
			break;
		default:
			NeoAssert( false );
	}
}

void CDnnBlob::TransferDataToThisThread()
{
	NeoAssert( dataOwned );
	NeoAssert( !data.IsNull() );
	NeoAssert( parent == nullptr );
	NeoAssert( GetDataType() == CT_Float || GetDataType() == CT_Int );

	const size_t size = GetDataSize()
		* ( ( GetDataType() == CT_Float ) ? sizeof( float ) : sizeof( int ) );
	mathEngine.TransferHandleToThisThread( data, size );
}

void CDnnBlob::Add(const CDnnBlob* other)
{
	NeoPresume(other->GetDataSize() == GetDataSize());
	switch(GetDataType()) {
		case CT_Float:
			mathEngine.VectorAdd( GetData<float>(), other->GetData<float>(), GetData<float>(), GetDataSize() );
			break;
		case CT_Int:
			mathEngine.VectorAdd( GetData<int>(), other->GetData<int>(), GetData<int>(), GetDataSize() );
			break;
		default:
			NeoAssert( false );
	}
}

void CDnnBlob::Clear()
{
	switch(GetDataType()) {
		case CT_Float:
			mathEngine.VectorFill( GetData<float>(), 0, GetDataSize() );
			break;
		case CT_Int:
			mathEngine.VectorFill( GetData<int>(), 0, GetDataSize() );
			break;
		default:
			NeoAssert( false );
	}
}

void CDnnBlob::ClearObject(int num)
{
	switch(GetDataType()) {
		case CT_Float:
			mathEngine.VectorFill(GetObjectData<float>( num ), 0, GetObjectSize());
			break;
		case CT_Int:
			mathEngine.VectorFill(GetObjectData<int>( num ), 0, GetObjectSize());
			break;
		default:
			NeoAssert( false );
	}
}

CDnnBlob* CDnnBlob::GetTransposed(int _d1, int _d2) const
{
	if( _d1 == _d2 ) {
		return GetCopy();
	}

	int d1 = min(_d1, _d2);
	int d2 = max(_d1, _d2);

	int height = DimSize(d1);
	int width = DimSize(d2);
	int batchSize = 1, medium = 1, channels = 1;
	for(int d = 0; d < d1; d++) {
		batchSize *= DimSize(d);
	}
	for(int d = d1 + 1; d < d2; d++) {
		medium *= DimSize(d);
	}
	for(int d = d2 + 1; d < CBlobDesc::MaxDimensions; d++) {
		channels *= DimSize(d);
	}
	CBlobDesc transposed = desc;
	transposed.SetDimSize(d1, width);
	transposed.SetDimSize(d2, height);

	CDnnBlob* result = FINE_DEBUG_NEW CDnnBlob( mathEngine );
	result->initializeByPattern(GetDataType(), transposed);
	switch(GetDataType()) {
		case CT_Float:
			mathEngine.TransposeMatrix(batchSize, GetData<float>(), height, medium,
				width, channels, result->GetData<float>(), result->GetDataSize());
			break;
		case CT_Int:
			mathEngine.TransposeMatrix(batchSize, GetData<int>(), height, medium,
				width, channels, result->GetData<int>(), result->GetDataSize());
			break;
		default:
			NeoAssert( false );
	}
	return result;
}

void CDnnBlob::TransposeFrom(const CDnnBlob* other, int _d1, int _d2)
{
	if( _d1 == _d2 ) {
		CopyFrom( other );
		return;
	}

	int d1 = min(_d1, _d2);
	int d2 = max(_d1, _d2);
	NeoAssert( GetDataType() == other->GetDataType() && GetDataSize() == other->GetDataSize() );
	
	int height = other->DimSize(d1);
	int width = other->DimSize(d2);
	NeoAssert(height == DimSize(d2) && width == DimSize(d1));

	int batchSize = 1, medium = 1, channels = 1;
	for(int d = 0; d < d1; d++) {
		batchSize *= other->DimSize(d);
	}
	for(int d = d1 + 1; d < d2; d++) {
		medium *= other->DimSize(d);
	}
	for(int d = d2 + 1; d < CBlobDesc::MaxDimensions; d++) {
		channels *= other->DimSize(d);
	}
	switch(GetDataType()) {
		case CT_Float:
			mathEngine.TransposeMatrix(batchSize, other->GetData<float>(), height, medium,
				width, channels, GetData<float>(), GetDataSize());
			break;
		case CT_Int:
			mathEngine.TransposeMatrix(batchSize, other->GetData<int>(), height, medium,
				width, channels, GetData<int>(), GetDataSize());
			break;
		default:
			NeoAssert( false );
	}
}

// Changes the blob dimensions "names" without moving the data
// In effect, only the blob descriptor is changed
// As the data is unaffected, the total blob size specified by the new descriptor should be the same
void CDnnBlob::ReinterpretDimensions( const CBlobDesc& newDesc )
{
	NeoAssert( parent == 0 );
	NeoAssert( newDesc.BlobSize() == desc.BlobSize() );

	desc = newDesc;
}

void CDnnBlob::MergeByDim( IMathEngine& mathEngine, TBlobDim d, const CObjectArray<CDnnBlob>& from, const CPtr<CDnnBlob>& to )
{
	CFastArray<CBlobDesc, 16> fromArray;
	fromArray.SetSize( from.Size() );
	if( to->GetDataType() == CT_Float ) {
		CFastArray<CFloatHandle, 16> fromData;
		fromData.SetSize( from.Size() );
		for( int i = 0; i < from.Size(); ++i ) {
			fromArray[i] = from[i]->GetDesc();
			fromData[i] = from[i]->GetData();
		}
		mathEngine.BlobMergeByDim( d, fromArray.GetPtr(), fromData.GetPtr(), from.Size(), to->GetDesc(), to->GetData() );
	} else {
		CFastArray<CIntHandle, 16> fromData;
		fromData.SetSize( from.Size() );
		for( int i = 0; i < from.Size(); ++i ) {
			fromArray[i] = from[i]->GetDesc();
			fromData[i] = from[i]->GetData<int>();
		}
		mathEngine.BlobMergeByDim( d, fromArray.GetPtr(), fromData.GetPtr(), from.Size(), to->GetDesc(), to->GetData<int>() );
	}
}

void CDnnBlob::MergeByChannels( IMathEngine& mathEngine, const CObjectArray<CDnnBlob>& from, const CPtr<CDnnBlob>& to )
{
	MergeByDim( mathEngine, BD_Channels, from, to );
}

void CDnnBlob::MergeByDepth( IMathEngine& mathEngine, const CObjectArray<CDnnBlob>& from, const CPtr<CDnnBlob>& to )
{
	MergeByDim( mathEngine, BD_Depth, from, to );
}

void CDnnBlob::MergeByWidth( IMathEngine& mathEngine, const CObjectArray<CDnnBlob>& from, const CPtr<CDnnBlob>& to )
{
	MergeByDim( mathEngine, BD_Width, from, to );
}

void CDnnBlob::MergeByHeight( IMathEngine& mathEngine, const CObjectArray<CDnnBlob>& from, const CPtr<CDnnBlob>& to )
{
	MergeByDim( mathEngine, BD_Height, from, to );
}

void CDnnBlob::MergeByListSize( IMathEngine& mathEngine, const CObjectArray<CDnnBlob>& from, const CPtr<CDnnBlob>& to )
{
	MergeByDim( mathEngine, BD_ListSize, from, to );
}

void CDnnBlob::MergeByBatchWidth( IMathEngine& mathEngine, const CObjectArray<CDnnBlob>& from, const CPtr<CDnnBlob>& to )
{
	MergeByDim( mathEngine, BD_BatchWidth, from, to );
}

void CDnnBlob::MergeByBatchLength( IMathEngine& mathEngine, const CObjectArray<CDnnBlob>& from, const CPtr<CDnnBlob>& to )
{
	MergeByDim( mathEngine, BD_BatchLength, from, to );
}

void CDnnBlob::MergeByObject( IMathEngine& mathEngine, const CObjectArray<CDnnBlob>& from, const CPtr<CDnnBlob>& to )
{
	MergeByDim( mathEngine, static_cast<TBlobDim>(CBlobDesc::FirstObjectDim), from, to );
}

void CDnnBlob::SplitByDim( IMathEngine& mathEngine, TBlobDim d, const CPtr<const CDnnBlob>& from, const CObjectArray<CDnnBlob>& to )
{
	CFastArray<CBlobDesc, 16> toArray;
	toArray.SetSize( to.Size() );
	if( from->GetDataType() == CT_Float ) {
		CFastArray<CFloatHandle, 16> toData;
		toData.SetSize( to.Size() );
		for( int i = 0; i < to.Size(); ++i ) {
			toArray[i] = to[i]->GetDesc();
			toData[i] = to[i]->GetData();
		}
		mathEngine.BlobSplitByDim( d, from->GetDesc(), from->GetData<const float>(), toArray.GetPtr(), toData.GetPtr(), to.Size() );
	} else {
		CFastArray<CIntHandle, 16> toData;
		toData.SetSize( to.Size() );
		for( int i = 0; i < to.Size(); ++i ) {
			toArray[i] = to[i]->GetDesc();
			toData[i] = to[i]->GetData<int>();
		}
		mathEngine.BlobSplitByDim( d, from->GetDesc(), from->GetData<const int>(), toArray.GetPtr(), toData.GetPtr(), to.Size() );
	}
}

void CDnnBlob::SplitByChannels( IMathEngine& mathEngine, const CPtr<const CDnnBlob>& from, const CObjectArray<CDnnBlob>& to )
{
	SplitByDim( mathEngine, BD_Channels, from, to );
}

void CDnnBlob::SplitByDepth( IMathEngine& mathEngine, const CPtr<const CDnnBlob>& from, const CObjectArray<CDnnBlob>& to )
{
	SplitByDim( mathEngine, BD_Depth, from, to );
}

void CDnnBlob::SplitByWidth( IMathEngine& mathEngine, const CPtr<const CDnnBlob>& from, const CObjectArray<CDnnBlob>& to )
{
	SplitByDim( mathEngine, BD_Width, from, to );
}

void CDnnBlob::SplitByHeight( IMathEngine& mathEngine, const CPtr<const CDnnBlob>& from, const CObjectArray<CDnnBlob>& to )
{
	SplitByDim( mathEngine, BD_Height, from, to );
}

void CDnnBlob::SplitByListSize( IMathEngine& mathEngine, const CPtr<const CDnnBlob>& from, const CObjectArray<CDnnBlob>& to )
{
	SplitByDim( mathEngine, BD_ListSize, from, to );
}

void CDnnBlob::SplitByBatchWidth( IMathEngine& mathEngine, const CPtr<const CDnnBlob>& from, const CObjectArray<CDnnBlob>& to )
{
	SplitByDim( mathEngine, BD_BatchWidth, from, to );
}

void CDnnBlob::SplitByBatchLength( IMathEngine& mathEngine, const CPtr<const CDnnBlob>& from, const CObjectArray<CDnnBlob>& to )
{
	SplitByDim( mathEngine, BD_BatchLength, from, to );
}

void CDnnBlob::SplitByObject( IMathEngine& mathEngine, const CPtr<const CDnnBlob>& from, const CObjectArray<CDnnBlob>& to )
{
	SplitByDim( mathEngine, static_cast<TBlobDim>(CBlobDesc::FirstObjectDim), from, to );
}

// Reads the data from archive into blob memory; in case of CPU reads directly
template<typename T>
static void readRawData( IMathEngine& mathEngine, CArchive& archive, const CTypedMemoryHandle<T>& handle )
{
	unsigned int size = 0;
	archive >> size;
	check( static_cast<int>( size ) >= 0, ERR_BAD_ARCHIVE, archive.Name() );

	if( size > 0 ) {
		void* ptr = mathEngine.GetBuffer( handle, 0, size * sizeof(T), false );
		archive.Read( ptr, size * sizeof(T) );
		mathEngine.ReleaseBuffer( handle, ptr, true );
	}
}

template<typename T>
static void writeRawData( IMathEngine& mathEngine, const int size, const CTypedMemoryHandle<T>& handle, CArchive& archive )
{
	archive << static_cast<unsigned int>( size );

	if( size > 0 ) {
		void* ptr = mathEngine.GetBuffer( handle, 0, size * sizeof(T), true );
		archive.Write( ptr, size * sizeof(T) );
		mathEngine.ReleaseBuffer( handle, ptr, false );
	}
}

static const int BlobVersion = 2000;

void CDnnBlob::Serialize( CArchive& archive )
{
	NeoAssert( parent == 0 ); // a blob that links to another may not be serialized

	archive.SerializeVersion( BlobVersion, CDnn::ArchiveMinSupportedVersion );

	if( archive.IsStoring() ) {
		archive << static_cast<int>( GetDataType() );
		archive << static_cast<int>( 0 );
		for( TBlobDim d = TBlobDim(0); d < BD_Count; ++d ) {
			archive << desc.DimSize(d);
		}

		switch(GetDataType()) {
			case CT_Float:
				writeRawData( mathEngine, desc.BlobSize(), GetData<float>(), archive );
				break;
			case CT_Int:
				writeRawData( mathEngine, desc.BlobSize(), GetData<int>(), archive );
				break;
			default:
				NeoAssert( false );
		}
	} else if( archive.IsLoading() ) {
		int intType;
		archive >> intType;
		TBlobType type = static_cast<TBlobType>(intType);

		int intPack;
		archive >> intPack;
		int batchLength, batchWidth, listSize, height, width, depth, channels;
		archive >> batchLength >> batchWidth >> listSize >> height >> width >> depth >> channels;
		initializeBlob(type, batchLength, batchWidth, listSize, height, width, depth, channels);

		switch( type ) {
			case CT_Float:
				readRawData( mathEngine, archive, GetData<float>() );
				break;
			case CT_Int:
				readRawData( mathEngine, archive, GetData<int>() );
				break;
			default:
				NeoAssert( false );
		}
		parentPos = 0;
	} else {
		NeoAssert( false );
	}
}

} // namespace NeoML
