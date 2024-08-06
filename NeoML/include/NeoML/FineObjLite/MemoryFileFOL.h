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

#include <AllocFOL.h>
#include <ErrorsFOL.h>
#include <BaseFileFOL.h>

namespace FObj {

const int MemoryFileDefaultGrowBytes = 1024;

// The class for working with in-memory files.
template<class Allocator = CurrentMemoryManager>
class CMemoryFileEx : public CBaseFile {
public:
	explicit CMemoryFileEx( int _growBytes = MemoryFileDefaultGrowBytes ) : growBytes( _growBytes )
	{ AssertFO( growBytes >= 0 ); }

	~CMemoryFileEx() override { Close(); }

	void MoveTo( CMemoryFileEx<Allocator>& other );

	bool IsOpen() const { return isOpen; }

	// Replace file buffer. Owner of the new buffer.
	// The new buffer should have the same Allocator
	void Attach( BYTE* buffer, size_t size, int newGrowBytes );
	// Detach buffer from file. Return the buffer pointer.
	// To delete returned buffer use Allocator
	BYTE* Detach();
	void DeleteBuffer();

	// CBaseFile methods:
	const char* GetFileName() const override { return "Memory file."; }
	int Read( void*, int bytesCount ) override;
	void Write( const void*, int bytesCount ) override;
	void Close() override;
	__int64 GetPosition() const override;
	__int64 Seek( __int64 offset, TSeekPosition from ) override;
	void SetLength( __int64 newLength ) override;
	__int64 GetLength() const override;
	void Abort() override;
	void Flush() override;

	// Direct access
	const BYTE* GetBufferPtr() const { return buffer; }
	size_t GetBufferSize() const { return bufferSize; }

protected:
	virtual void FreeBuffer( BYTE* );
	virtual BYTE* GrowBuffer( BYTE*, size_t oldSize, size_t newSize );

private:
	BYTE* buffer = nullptr; // file buffer
	size_t bufferSize = 0; // current buffer size
	size_t fileLength = 0; // file size
	int growBytes = 0; // minimum increment of buffer size
	size_t currentPosition = 0; // current offset of the file pointer
	bool isOpen = true; // whether file is open

	void setBufferSize( size_t requiredSize );
	void throwBadSeekException();
};

//---------------------------------------------------------------------------------------------------------------------

class CMemoryFile : public CMemoryFileEx<> {
public:
	CMemoryFile( int growBytes = MemoryFileDefaultGrowBytes ) : CMemoryFileEx<>( growBytes ) {}
};

//---------------------------------------------------------------------------------------------------------------------

template<class Allocator>
inline void CMemoryFileEx<Allocator>::Attach( BYTE* newBuffer, size_t size, int newGrowBytes )
{
	AssertFO( newBuffer != nullptr );
	AssertFO( size > 0 );
	AssertFO( newGrowBytes >= 0 );

	if( buffer != nullptr ) {
		PresumeFO( newBuffer != buffer );
		FreeBuffer( buffer );
	}
	buffer = newBuffer;
	bufferSize = size;
	growBytes = newGrowBytes;
	fileLength = 0;
	currentPosition = 0;
}

template<class Allocator>
inline BYTE* CMemoryFileEx<Allocator>::Detach()
{
	BYTE* ret = buffer;
	buffer = nullptr;
	bufferSize = 0;
	fileLength = 0;
	currentPosition = 0;
	return ret;
}

template<class Allocator>
inline void CMemoryFileEx<Allocator>::DeleteBuffer()
{
	if( buffer != nullptr ) {
		FreeBuffer( buffer );
	}
	buffer = nullptr;
	bufferSize = 0;
	fileLength = 0;
	currentPosition = 0;
}

template<class Allocator>
inline void CMemoryFileEx<Allocator>::MoveTo( CMemoryFileEx<Allocator>& other )
{
	BYTE* otherBuffer = other.Detach();
	if( otherBuffer != 0 ) {
		FreeBuffer( otherBuffer );
	}

	other.fileLength = fileLength;
	other.bufferSize = bufferSize;
	other.buffer = Detach();
}

template<class Allocator>
inline int CMemoryFileEx<Allocator>::Read( void* ptr, int bytesCount )
{
	if( bytesCount == 0 || fileLength <= currentPosition ) {
		return 0;
	}
	AssertFO( ptr != 0 );
	AssertFO( bytesCount > 0 );

	const size_t size = min( fileLength - currentPosition, static_cast<size_t>( bytesCount ) );
	::memcpy( ptr, buffer + currentPosition, size );
	currentPosition += size;
	return static_cast<int>( size );
}

template<class Allocator>
inline void CMemoryFileEx<Allocator>::Write( const void* ptr, int bytesCount )
{
	if( bytesCount == 0 ) {
		return;
	}

	AssertFO( ptr != 0 );
	AssertFO( bytesCount > 0 );

	auto newPosition = currentPosition + bytesCount;
	if( newPosition > bufferSize ) {
		setBufferSize( newPosition );
	}
	::memcpy( buffer + currentPosition, ptr, bytesCount );
	currentPosition = newPosition;
	fileLength = max( fileLength, currentPosition );
}

template<class Allocator>
inline void CMemoryFileEx<Allocator>::Close()
{
	if( !IsOpen() ) {
		return;
	}

	if( buffer != 0 ) {
		FreeBuffer( buffer );
	}
	buffer = 0;
	bufferSize = 0;
	fileLength = 0;
	currentPosition = 0;
	isOpen = false;
}

template<class Allocator>
inline __int64 CMemoryFileEx<Allocator>::GetPosition() const
{
	return currentPosition;
}

template<class Allocator>
inline __int64 CMemoryFileEx<Allocator>::Seek( __int64 offset, TSeekPosition from )
{
	__int64 newPosition = currentPosition;
	switch( from ) {
		case current:
			newPosition += offset;
			break;
		case begin:
			newPosition = offset;
			break;
		case end:
			newPosition = GetLength() + offset;
			break;
		default:
			AssertFO( false );
	}
	if( 0 <= newPosition && newPosition <= LONG_MAX ) {
		currentPosition = static_cast<size_t>( newPosition );
	} else {
		currentPosition = 0;
		throwBadSeekException();
	}
	return currentPosition;
}

template<class Allocator>
inline void CMemoryFileEx<Allocator>::SetLength( __int64 newLength )
{
	AssertFO( 0 <= newLength && newLength <= LONG_MAX );
	size_t newSize = static_cast<size_t>( newLength );
	if( bufferSize < newSize ) {
		setBufferSize( newSize );
	}
	if( newSize < currentPosition ) {
		currentPosition = newSize;
	}
	fileLength = newSize;
}

template<class Allocator>
inline __int64 CMemoryFileEx<Allocator>::GetLength() const
{
	return fileLength;
}

template<class Allocator>
inline void CMemoryFileEx<Allocator>::Abort()
{
	Close();
}

template<class Allocator>
inline void CMemoryFileEx<Allocator>::Flush()
{
}

template<class Allocator>
inline void CMemoryFileEx<Allocator>::FreeBuffer( BYTE* ptr )
{
	Allocator::Free( ptr );
}

template<class Allocator>
inline BYTE* CMemoryFileEx<Allocator>::GrowBuffer( BYTE* oldBuffer, size_t oldSize, size_t newSize )
{
	AssertFO( newSize > oldSize );
	BYTE* newBuffer = static_cast<BYTE*>( Allocator::Alloc( newSize * sizeof( BYTE ) ) );
	if( oldSize > 0 ) {
		::memcpy( newBuffer, oldBuffer, oldSize );
	}
	if( oldBuffer != 0 ) {
		FreeBuffer( oldBuffer );
	}
	return newBuffer;
}

template<class Allocator>
inline void CMemoryFileEx<Allocator>::setBufferSize( size_t requiredSize )
{
	AssertFO( growBytes > 0 );
	// Exponential enlargement, newBufferSize >= requiredSize
	int newSizeDiff = static_cast<int>( requiredSize - bufferSize );
	size_t newBufferSize = bufferSize + max( bufferSize / 2, static_cast<size_t>( CeilTo( newSizeDiff, growBytes ) ) );
	buffer = GrowBuffer( buffer, bufferSize, newBufferSize );
	AssertFO( buffer != 0 );
	bufferSize = newBufferSize;
}

template<class Allocator>
inline void CMemoryFileEx<Allocator>::throwBadSeekException()
{
#if FINE_PLATFORM( FINE_WINDOWS )
	ThrowFileException( ERROR_SEEK, GetFileName() );
#elif FINE_PLATFORM( FINE_ANDROID ) || FINE_PLATFORM( FINE_IOS ) || FINE_PLATFORM( FINE_LINUX ) || FINE_PLATFORM( FINE_DARWIN )
	ThrowFileException( EOVERFLOW, GetFileName() );
#else
	#error "Platform doesn't supported!"
#endif
}

} // namespace FObj
