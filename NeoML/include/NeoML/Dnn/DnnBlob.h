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

#pragma once

#include <NeoML/NeoMLDefs.h>
#include <NeoMathEngine/MemoryHandle.h>
#include <NeoMathEngine/NeoMathEngine.h>

namespace NeoML {

// CDnnBlob is a block of data transmitted into and out of the network and between layers
// CDnnBlob is represented by a 7-dimensional tensor, with the dimensions standing for:
// BatchLength - the sequence length
// BatchWidth - the number of sequences or elements to be processed together
// ListSize - the length of the list to be processed together
// H - Height - the image height
// W - Width - the image width
// D - Depth - the image depth (for 3D images)
// C - Channels - corresponds to "channels" of the object; they may be color channels for an image
//		 or have some other purpose, for example, be used as convolution filters or a simple layer neurons

class NEOML_API CDnnBlob : public IObject {
public:
	explicit CDnnBlob( IMathEngine& mathEngine ) : mathEngine( mathEngine ) {}

	// Move other's Blob state to this Blob and transfer its data to this thread
	CDnnBlob( CDnnBlob&& other );
	CDnnBlob& operator=( CDnnBlob&& other );

	// Create blobs of various kinds
	static CDnnBlob* CreateVector(IMathEngine& mathEngine, TBlobType type, int vectorSize);
	static CDnnBlob* CreateMatrix(IMathEngine& mathEngine, TBlobType type, int matrixHeight, int matrixWidth);
	static CDnnBlob* CreateTensor(IMathEngine& mathEngine, TBlobType type, std::initializer_list<int> dimensions);

	// Creates a blob with numerical data
	static CDnnBlob* CreateDataBlob(IMathEngine& mathEngine, TBlobType type, int batchLength, int batchWidth,
		int channelsCount);
	// Creates a blob with object lists (that is, one-dimensional data)
	static CDnnBlob* CreateListBlob( IMathEngine& mathEngine,
		TBlobType type, int batchLength, int batchWidth, int listSize, int channelsCount );
	// Creates a blob with two-dimensional images
	static CDnnBlob* Create2DImageBlob(IMathEngine& mathEngine, TBlobType type, int batchLength, int batchWidth,
		int imageHeight, int imageWidth, int channelsCount);
	// Creates a blob with 3-dimensional images
	static CDnnBlob* Create3DImageBlob( IMathEngine& mathEngine,
		TBlobType type, int batchLength, int batchWidth, 
		int imageHeight, int imageWidth, int imageDepth, int channelsCount );
	// Creates a "window" blob to represent a subsequence of objects from the parent blob
	static CDnnBlob* CreateWindowBlob(const CPtr<CDnnBlob>& parent, int windowSize = 1);
	// Creates a blob according to the provided descriptor
	static CDnnBlob* CreateBlob(IMathEngine& mathEngine, const CBlobDesc& pattern);
	static CDnnBlob* CreateBlob(IMathEngine& mathEngine, TBlobType type, const CBlobDesc& pattern);

	// Checks if the dimensions of another blob are the same
	bool HasEqualDimensions( const CDnnBlob* other ) const { return desc.HasEqualDimensions( other->desc ); }

	// Gets the blob size along the specified dimension
	int DimSize(int d) const { return desc.DimSize(d); }
	int DimSize(TBlobDim d) const { return desc.DimSize(d); }

	int GetBatchLength() const { return desc.BatchLength(); }
	int GetBatchWidth() const { return desc.BatchWidth(); }
	int GetListSize() const { return desc.ListSize(); }
	int GetObjectCount() const { return desc.ObjectCount(); }
	int GetHeight() const { return desc.Height(); }
	int GetWidth() const { return desc.Width(); }
	int GetDepth() const { return desc.Depth(); }
	int GetChannelsCount() const { return desc.Channels(); }

	// Gets the size of data in the blob
	int GetDataSize() const { return desc.BlobSize(); }
	int GetObjectSize() const { return desc.ObjectSize(); }
	// Gets the geometrical size of the blob (Height * Width * Depth)
	int GetGeometricalSize() const { return desc.GeometricalSize(); }

	// Gets the handle to the data
	template<class T = float>
	CTypedMemoryHandle<const T> GetData() const;
	template<class T = float>
	CTypedMemoryHandle<T> GetData();
	template<class T = float>
	CTypedMemoryHandle<const T> GetData( std::initializer_list<int> position ) const;
	template<class T = float>
	CTypedMemoryHandle<T> GetData( std::initializer_list<int> position );

	template<class T = float>
	CTypedMemoryHandle<const T> GetObjectData( int objectNum ) const;
	template<class T = float>
	CTypedMemoryHandle<T> GetObjectData( int objectNum );

	// Data exchange
	template<class T = float>
	void CopyFrom(const T* src);

	template<class T = float>
	void CopyTo(T* dst, int size) const;

	// Copies the entire contents of the blob into memory
	template<class T = float>
	void CopyTo(T* dst) const;

	// Next functions provide access to the blob memory through pointers
	// The GetBuffer and ReleaseBuffer methods should be called strictly on the last-in-first-out principle within the same thread

	// Returns pointer to the blob memory
	// pos sets the position in the blob where the block of memory will start
	// size sets the size of the block of memory
	// If pos + size > the total blob size, 0 will be returned
	// If exchange == true then it's guaranteed that pointer will contain actual blob data
	// (otherwise it's implementation-dependent)
	// It's recommended to use exchange == false when you need a write-only buffer
	template<class T>
	T* GetBuffer( int pos, int size, bool exchange );

	// Releases previously created buffer to the blob memory
	// If exchange == true then it's guaranteed that changes in the buffer will take place in the blob
	// (otherwise it's implementation-dependent)
	// It's recommended to use exchange == false when you need a read-only buffer
	void ReleaseBuffer( void* ptr, bool exchange );

	// Creates an empty blob of the same dimensions
	CDnnBlob* GetClone() const;
	CDnnBlob* GetClone(TBlobType _type) const;

	// Copies the blob
	CDnnBlob* GetCopy() const;
	// Copies the contents from another blob
	void CopyFrom(const CDnnBlob* other);

	// Transfers CDnnBlob data from other thread owner to this thread.
	// By default memory underneath each blob is associated with the thread on which its allocation has occurred.
	// This method switches this association to the calling thread.
	virtual void TransferDataToThisThread();

	// Elementwise adds a blob of the same dimensions
	void Add(const CDnnBlob* other);
	// Clears the contents
	void Clear();
	void ClearObject( int num );
	// Fills the blob with the given value 
	// (CBlobType<T>::TDataType is used to avoid invalid typecasts when a call like Fill(0) is made)
	template<class T = float>
	void Fill(typename CBlobType<T>::TDataType value);
	template<class T = float>
	void FillObject(int num, typename CBlobType<T>::TDataType value);

	// Transposes two dimensions of the blob; the data in memory is moved
	CDnnBlob* GetTransposed(int d1, int d2) const;
	// Transposes the data from another blob and copies it to this one
	void TransposeFrom(const CDnnBlob* other, int d1, int d2);

	// Changes the blob dimensions "names" without moving the data
	// In effect, only the blob description is changed
	// As the data is unaffected, the total blob size specified by the new descriptor should be the same
	virtual void ReinterpretDimensions( const CBlobDesc& newDesc );

	// Merges blobs along the given dimension
	static void MergeByDim( IMathEngine& mathEngine, TBlobDim d, const CObjectArray<CDnnBlob>& from, const CPtr<CDnnBlob>& to );
	static void MergeByChannels( IMathEngine& mathEngine, const CObjectArray<CDnnBlob>& from, const CPtr<CDnnBlob>& to );
	static void MergeByDepth( IMathEngine& mathEngine, const CObjectArray<CDnnBlob>& from, const CPtr<CDnnBlob>& to );
	static void MergeByWidth( IMathEngine& mathEngine, const CObjectArray<CDnnBlob>& from, const CPtr<CDnnBlob>& to );
	static void MergeByHeight( IMathEngine& mathEngine, const CObjectArray<CDnnBlob>& from, const CPtr<CDnnBlob>& to );
	static void MergeByListSize( IMathEngine& mathEngine, const CObjectArray<CDnnBlob>& from, const CPtr<CDnnBlob>& to );
	static void MergeByBatchWidth( IMathEngine& mathEngine, const CObjectArray<CDnnBlob>& from, const CPtr<CDnnBlob>& to );
	static void MergeByBatchLength( IMathEngine& mathEngine, const CObjectArray<CDnnBlob>& from, const CPtr<CDnnBlob>& to );
	static void MergeByObject( IMathEngine& mathEngine, const CObjectArray<CDnnBlob>& from, const CPtr<CDnnBlob>& to );

	// Splits blobs along the given dimension
	static void SplitByDim( IMathEngine& mathEngine, TBlobDim d, const CPtr<const CDnnBlob>& from, const CObjectArray<CDnnBlob>& to );
	static void SplitByChannels( IMathEngine& mathEngine, const CPtr<const CDnnBlob>& from, const CObjectArray<CDnnBlob>& to );
	static void SplitByDepth( IMathEngine& mathEngine, const CPtr<const CDnnBlob>& from, const CObjectArray<CDnnBlob>& to );
	static void SplitByWidth( IMathEngine& mathEngine, const CPtr<const CDnnBlob>& from, const CObjectArray<CDnnBlob>& to );
	static void SplitByHeight( IMathEngine& mathEngine, const CPtr<const CDnnBlob>& from, const CObjectArray<CDnnBlob>& to );
	static void SplitByListSize( IMathEngine& mathEngine, const CPtr<const CDnnBlob>& from, const CObjectArray<CDnnBlob>& to );
	static void SplitByBatchWidth( IMathEngine& mathEngine, const CPtr<const CDnnBlob>& from, const CObjectArray<CDnnBlob>& to );
	static void SplitByBatchLength( IMathEngine& mathEngine, const CPtr<const CDnnBlob>& from, const CObjectArray<CDnnBlob>& to );
	static void SplitByObject( IMathEngine& mathEngine, const CPtr<const CDnnBlob>& from, const CObjectArray<CDnnBlob>& to );

	void Serialize( CArchive& ) override;

	// Gets the pointer to the MathEngine on which the blob was created
	IMathEngine& GetMathEngine() const { return mathEngine; }
	// Gets the blob descriptor
	const CBlobDesc& GetDesc() const { return desc; }
	// Gets the type of data in the blob
	TBlobType GetDataType() const { return desc.GetDataType(); }

	// All methods below are just the interface for the CDnnWindowBlob representation

	// Gets the parent blob
	virtual CDnnBlob* GetParent() { return nullptr; }
	const CDnnBlob* GetParent() const { return const_cast<CDnnBlob*>( this )->GetParent(); }
	// Gets the blob that owns the data (and has no parent)
	virtual CDnnBlob* GetOwner() { return this; }
	const CDnnBlob* GetOwner() const { return const_cast<CDnnBlob*>( this )->GetOwner(); }
	// Gets the shift in data relative to the parent blob
	// The position in the parent blob is calculated along the BatchLength dimension
	// The position equal to N would correspond to a N*BatchWidth*ListSize*Height*Width*Depth*Channels shift in the one-dimensional array
	virtual int GetParentPos() const { return 0; }
	virtual void SetParentPos( int /*pos*/ ) { NeoAssert( false ); }
	virtual void ShiftParentPos( int /*shift*/ ) { NeoAssert( false ); }

protected:
	~CDnnBlob() override;

	CDnnBlob( IMathEngine& _mathEngine, const CBlobDesc& _desc, CMemoryHandle _data ) :
		mathEngine( _mathEngine ), desc( _desc ), data( _data )
	{
		NeoAssert( desc.GetDataType() != CT_Invalid );
		NeoAssert( &mathEngine == data.GetMathEngine() );
	}

private:
	// Math Engine owner
	IMathEngine& mathEngine;
	// Actual typed sizes description of the allocated data storage
	CBlobDesc desc;
	// Pointer to the allocated data storage
	CMemoryHandle data;

	void initializeBlob(TBlobType _type, int batchLength, int batchWidth, int listSize, int height, int width,
		int depth, int channels);
	void initializeTensor(TBlobType _type, std::initializer_list<int> dimensions);
	void initializeByPattern(TBlobType type, const CBlobDesc& pattern);

	friend class CDnnBlobClassRegistrar;
	friend class CDnnWindowBlob;
	friend class CDnnBlobView;
};

//---------------------------------------------------------------------------------------------------------------------

// The kind of CDnnBlob does not own the data
// CDnnBlobView does not clear data handler
// Used in python wrappers
class NEOML_API CDnnBlobView : public CDnnBlob {
protected:
	CDnnBlobView( IMathEngine& mathEngine ) : CDnnBlob( mathEngine ) {}
	CDnnBlobView( IMathEngine& mathEngine, const CBlobDesc& desc, CMemoryHandle data ) :
		CDnnBlob( mathEngine, desc, data )
	{}

	~CDnnBlobView() { if( !data.IsNull() ) { data = CMemoryHandle{}; } } // no need to free

	// Prohibited methods
	//void ReinterpretDimensions( const CBlobDesc& ) override { NeoAssert( false ); } // impossible !!! USED in Python !!!
	//void Serialize( CArchive& ) override // !!! PICKLED in Python !!!
	//{ NeoAssert( false ); } // a blob that links to another may not be serialized
	void TransferDataToThisThread() override {} // !!! MOVED in Python !!!
};

//---------------------------------------------------------------------------------------------------------------------

// The kind of CDnnBlob that is view of the some parent Blob as sequence (BatchLength > 1).
// This CDnnWindowBlob do not owner of its memory, technical CDnnBlob representation.
// This CDnnWindowBlob represents 1 element (BatchLength == 1) of the sequence for the recursive networks.
class NEOML_API CDnnWindowBlob : public CDnnBlob {
public:
	// Creates a "window" blob to represent a subsequence of objects from the parent blob
	static CDnnBlob* CreateWindowBlob( const CPtr<CDnnBlob>& parent, int windowSize = 1 );

	// Prohibited methods
	void ReinterpretDimensions( const CBlobDesc& ) override { NeoAssert( false ); } // impossible
	void Serialize( CArchive& ) override { NeoAssert( false ); } // a blob that links to another may not be serialized
	void TransferDataToThisThread() override { NeoAssert( false ); }

	// Interface of communication
	CDnnBlob* GetParent() override { return parent; }
	CDnnBlob* GetOwner() override;
	int GetParentPos() const override;
	void SetParentPos( int pos ) override;
	void ShiftParentPos( int shift ) override;

protected:
	CDnnWindowBlob( IMathEngine& mathEngine ) : CDnnBlob( mathEngine ) {}
	CDnnWindowBlob( IMathEngine& mathEngine, const CBlobDesc& desc, CMemoryHandle data ) :
		CDnnBlob( mathEngine, desc, data )
	{}
	CDnnWindowBlob( CDnnWindowBlob&& other ) = delete;
	CDnnWindowBlob& operator=( CDnnWindowBlob&& other ) = delete;

	~CDnnWindowBlob() { if( parent != nullptr ) { data = CMemoryHandle{}; } } // no need to free

private:
	// Pointer to blob with data for sequential recurent mode or reference dnn's paramBlobs
	CPtr<CDnnBlob> parent;
	// Offset in `parent` blob for sequential recurent mode, move window by BatchLength of the parent blob
	int parentPos = 0;

	void initializeWindow( const CPtr<CDnnBlob>& parent, int windowSize );
};

//---------------------------------------------------------------------------------------------------------------------

inline void SerializeBlob( IMathEngine& mathEngine, CArchive& archive, CPtr<CDnnBlob>& blob )
{
	if( archive.IsStoring() ) {
		bool isNull = ( blob == 0 );
		archive << isNull;
		if( !isNull ) {
			blob->Serialize( archive );
		}
	} else if( archive.IsLoading() ) {
		bool isNull = false;
		archive >> isNull;
		if( isNull ) {
			blob = 0;
		} else {
			blob = FINE_DEBUG_NEW CDnnBlob( mathEngine );
			blob->Serialize( archive );
		}
	} else {
		NeoAssert( false );
	}
}

inline void SerializeBlobs( IMathEngine& mathEngine, CArchive& archive, CObjectArray<CDnnBlob>& blobs )
{
	if( archive.IsStoring() ) {
		archive << blobs.Size();
	} else if( archive.IsLoading() ) {
		int size = 0;
		archive >> size;
		blobs.SetSize( size );
	} else {
		NeoAssert( false );
	}

	for( int i = 0; i < blobs.Size(); i++ ) {
		SerializeBlob( mathEngine, archive, blobs[i] );
	}
}

//---------------------------------------------------------------------------------------------------------------------

enum class TDnnBlobBufferAccess {
	Read,
	Write,
	ReadWrite
};

// RAII-helper to safely work with `CDnnBlob::GetBuffer`/`CDnnBlob::ReleaseBuffer`.
template<typename TBufferType = float>
class CDnnBlobBuffer {
public:
	CDnnBlobBuffer( CDnnBlob& _blob, int pos, int _size, TDnnBlobBufferAccess _access ) :
		blob( _blob ),
		access( _access ),
		size( _size ),
		ptr( blob.GetBuffer<TBufferType>( pos, size, access == TDnnBlobBufferAccess::Read || access == TDnnBlobBufferAccess::ReadWrite ) )
	{}
	CDnnBlobBuffer( CDnnBlob& _blob, TDnnBlobBufferAccess _access ) :
		CDnnBlobBuffer( _blob, 0, _blob.GetDataSize(), _access )
	{}
	CDnnBlobBuffer( const CDnnBlobBuffer& ) = delete;
	~CDnnBlobBuffer();

	int Size() const { return size; }

	TBufferType* Ptr() { return ptr; }
	const TBufferType* Ptr() const { return ptr; }

	operator TBufferType*() { return ptr; }
	operator const TBufferType*() const { return ptr; }

	TBufferType& operator[]( int i ) { NeoAssert( !IsClosed() ); NeoPresume( 0 <= i && i < size ); return ptr[i]; }
	TBufferType operator[]( int i ) const { NeoAssert( !IsClosed() ); NeoPresume( 0 <= i && i < size ); return ptr[i]; }

	CDnnBlobBuffer& operator=( const CDnnBlobBuffer& ) = delete;

	// Explicitly close (and flush if requested) buffer. It is not possible to read/write data after.
	void Close();
	bool IsClosed() const { return ptr == nullptr; }

private:
	CDnnBlob& blob;
	TDnnBlobBufferAccess access;
	int size;
	TBufferType* ptr;
};

//---------------------------------------------------------------------------------------------------------------------

inline CDnnBlob* CDnnBlob::CreateBlob( IMathEngine& mathEngine, const CBlobDesc& pattern )
{
	return CreateBlob(mathEngine, CT_Float, pattern);
}

template<class T>
inline void CDnnBlob::Fill(typename CBlobType<T>::TDataType value)
{
	NeoAssert(GetDataType() == CBlobType<T>::GetType());
	mathEngine.VectorFill(GetData<T>(), value, GetDataSize());
}

template<class T>
inline void CDnnBlob::FillObject(int num, typename CBlobType<T>::TDataType value)
{
	NeoAssert(GetDataType() == CBlobType<T>::GetType());
	mathEngine.VectorFill(GetObjectData<T>(num), value, GetObjectSize());
}

template<class T>
inline CTypedMemoryHandle<const T> CDnnBlob::GetData() const
{
	NeoAssert(GetDataType() == CBlobType<T>::GetType());
	return CTypedMemoryHandle<const T>( data );
}

template<class T>
inline CTypedMemoryHandle<T> CDnnBlob::GetData()
{
	NeoAssert(GetDataType() == CBlobType<T>::GetType());
	return CTypedMemoryHandle<T>( data );
}

template<class T>
inline CTypedMemoryHandle<const T> CDnnBlob::GetData( std::initializer_list<int> position ) const
{
	NeoAssert(GetDataType() == CBlobType<T>::GetType());
	NeoAssert(position.size() <= CBlobDesc::MaxDimensions);

	int dataPos = 0;
	for(int i = 0; i < static_cast<int>(position.size()); i++) {
		dataPos *= DimSize(i);
		dataPos += position.begin()[i];
	}
	for(int i = static_cast<int>(position.size()); i < CBlobDesc::MaxDimensions; i++) {
		dataPos *= DimSize(i);
	}
	NeoAssert(dataPos < GetDataSize());
	return GetData<T>() + dataPos;
}

template<class T>
inline CTypedMemoryHandle<T> CDnnBlob::GetData( std::initializer_list<int> position )
{
	NeoAssert(GetDataType() == CBlobType<T>::GetType());
	NeoAssert(position.size() <= CBlobDesc::MaxDimensions);

	int dataPos = 0;
	for(int i = 0; i < static_cast<int>(position.size()); i++) {
		dataPos *= DimSize(i);
		dataPos += position.begin()[i];
	}
	for(int i = static_cast<int>(position.size()); i < CBlobDesc::MaxDimensions; i++) {
		dataPos *= DimSize(i);
	}
	NeoAssert(dataPos < GetDataSize());
	return GetData<T>() + dataPos;
}

template<class T>
inline CTypedMemoryHandle<const T> CDnnBlob::GetObjectData( int objectNum ) const
{
	NeoAssert(0 <= objectNum && objectNum < desc.ObjectCount());
	return GetData<T>() + objectNum * GetObjectSize();
}

template<class T>
inline CTypedMemoryHandle<T> CDnnBlob::GetObjectData( int objectNum )
{
	NeoAssert(0 <= objectNum && objectNum < desc.ObjectCount());
	return GetData<T>() + objectNum * GetObjectSize();
}

template<class T>
inline void CDnnBlob::CopyFrom(const T* src)
{
	mathEngine.DataExchangeRaw(GetData<T>(), src, GetDataSize() * sizeof(T));
}

template<class T>
inline void CDnnBlob::CopyTo(T* dst, int size) const
{
	mathEngine.DataExchangeRaw(dst, GetData<T>(), size * sizeof(T));
}

template<class T>
inline void CDnnBlob::CopyTo(T* dst) const
{
	CopyTo(dst, GetDataSize());
}

template<class T>
inline T* CDnnBlob::GetBuffer( int pos, int size, bool exchange )
{
	if( pos < 0 || GetDataSize() < pos + size ) {
		return 0;
	}

	int dataSize = 0;
	switch( desc.GetDataType() ) {
		case CT_Float:
			dataSize = sizeof( float );
			break;
		case CT_Int:
			dataSize = sizeof( int );
			break;
		default:
			NeoAssert( false );
	}

	return static_cast<T*>( mathEngine.GetBuffer( data, pos * dataSize, size * dataSize, exchange ) );
}

//---------------------------------------------------------------------------------------------------------------------

inline int CDnnWindowBlob::GetParentPos() const
{
	NeoAssert(parent != 0);
	return parentPos;
}

inline void CDnnWindowBlob::SetParentPos(int pos)
{
	int arrayPos = pos * (desc.BlobSize() / desc.BatchLength());
	NeoAssert(parent != 0);
	NeoAssert(arrayPos + desc.BlobSize() <= parent->desc.BlobSize());
	parentPos = pos;
	switch(GetDataType()) {
		case CT_Float:
			data = parent->GetData<float>() + arrayPos;
			break;
		case CT_Int:
			data = parent->GetData<int>() + arrayPos;
			break;
		default:
			NeoAssert(0);
	}
}

inline void CDnnWindowBlob::ShiftParentPos(int shift)
{
	SetParentPos(parentPos + shift);
}

inline CDnnBlob* CDnnWindowBlob::GetOwner()
{
	CDnnBlob* result = this;
	CDnnWindowBlob* window = this;
	while( window != nullptr && window->parent != nullptr ) {
		result = window->parent;
		window = dynamic_cast<CDnnWindowBlob*>( result );
	}
	return result;
}

//---------------------------------------------------------------------------------------------------------------------

template<typename TBufferType>
inline CDnnBlobBuffer<TBufferType>::~CDnnBlobBuffer()
{
	try {
		if( !IsClosed() ) {
			Close();
		}
#ifdef NEOML_USE_FINEOBJ
	} catch( CException* e ) {
		FineBreakPoint();
		delete e;
#else
	} catch( CException& e ) {
		(void) e;
		FineBreakPoint();
#endif
	}
}

template<typename TBufferType>
inline void CDnnBlobBuffer<TBufferType>::Close()
{
	NeoAssert( !IsClosed() );
	blob.ReleaseBuffer( ptr, access == TDnnBlobBufferAccess::Write || access == TDnnBlobBufferAccess::ReadWrite );
	ptr = nullptr;
}

} // namespace NeoML
