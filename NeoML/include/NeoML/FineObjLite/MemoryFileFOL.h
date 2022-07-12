/* Copyright © 2017-2020 ABBYY Production LLC

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

#include "AllocFOL.h"
#include "ErrorsFOL.h"
#include "BaseFileFOL.h"

namespace FObj {

const int MemoryFileDefaultGrowBytes = 1024;

// The class for working with in-memory files.
template<class Allocator = CurrentMemoryManager>
class CMemoryFileEx : public CBaseFile {
public:
	explicit CMemoryFileEx( int growBytes = MemoryFileDefaultGrowBytes );
	~CMemoryFileEx() override;

	bool IsOpen() const { return isOpen; }

	// CBaseFile methods:
	const char* GetFileName() const override { return "Memory file."; }
	int Read( void*, int bytesCount ) override;
	void Write( const void*, int bytesCount ) override;
	void Close() override;
	__int64 GetPosition() const override;
	__int64 Seek( __int64 offset, CBaseFile::TSeekPosition from ) override;
	void SetLength( __int64 newLength ) override;
	__int64 GetLength() const override;
	void Abort() override;
	void Flush() override;

protected:
	virtual void FreeBuffer( BYTE* );
	virtual BYTE* GrowBuffer( BYTE*, int oldSize, int newSize );

private:
	BYTE* buffer; // file buffer
	int bufferSize; // current buffer size
	int fileLength; // file size
	int growBytes; // minimum increment of buffer size
	int currentPosition; // current offset of the file pointer
	bool isOpen; // whether file is open

	void setBufferSize( int requiredSize );
	void throwBadSeekException();
};

class CMemoryFile : public CMemoryFileEx<> {
public:
	CMemoryFile(int growBytes = MemoryFileDefaultGrowBytes) : CMemoryFileEx<>(growBytes)
	{
	}
};

template<class Allocator>
inline CMemoryFileEx<Allocator>::CMemoryFileEx( int _growBytes ) :
	buffer( 0 ),
	bufferSize( 0 ),
	fileLength( 0 ),
	growBytes( _growBytes ),
	currentPosition( 0 ),
	isOpen( true )
{
	AssertFO( growBytes >= 0 );
}

template<class Allocator>
inline CMemoryFileEx<Allocator>::~CMemoryFileEx()
{
	Close();
}

template<class Allocator>
inline int CMemoryFileEx<Allocator>::Read( void* ptr, int bytesCount )
{
	if( bytesCount == 0 ) {
		return 0;
	}
	AssertFO( ptr != 0 );
	AssertFO( bytesCount > 0 );

	int size = min( fileLength - currentPosition, bytesCount );
	if( size <= 0 ) {
		return 0;
	}
	::memcpy( ptr, buffer + currentPosition, size );
	currentPosition += size;
	return size;
}

template<class Allocator>
inline void CMemoryFileEx<Allocator>::Write( const void* ptr, int bytesCount )
{
	if( bytesCount == 0 ) {
		return;
	}

	AssertFO( ptr != 0 );
	AssertFO( bytesCount > 0 );

	int newPosition = currentPosition + bytesCount;
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
	if( 0 <= newPosition && newPosition <= INT_MAX ) {
		currentPosition = to<int>( newPosition );
	} else {
		currentPosition = 0;
		throwBadSeekException();
	}
	return currentPosition;
}

template<class Allocator>
inline void CMemoryFileEx<Allocator>::SetLength( __int64 newLength )
{
	AssertFO( 0 <= newLength && newLength <= INT_MAX );
	int length32 = to<int>( newLength );
	if( bufferSize < length32 ) {
		setBufferSize( length32 );
	}
	if( length32 < currentPosition ) {
		currentPosition = length32;
	}
	fileLength = length32;
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
	CurrentMemoryManager::Free( ptr );
}

template<class Allocator>
inline BYTE* CMemoryFileEx<Allocator>::GrowBuffer( BYTE* oldBuffer, int oldSize, int newSize )
{
	AssertFO( newSize > oldSize );
	BYTE* newBuffer = static_cast<BYTE*>( ALLOCATE_MEMORY( CurrentMemoryManager, newSize * sizeof( BYTE ) ) );
	if( oldSize > 0 ) {
		::memcpy( newBuffer, oldBuffer, oldSize );
	}
	if( oldBuffer != 0 ) {
		FreeBuffer( oldBuffer );
	}
	return newBuffer;
}

template<class Allocator>
inline void CMemoryFileEx<Allocator>::setBufferSize( int requiredSize )
{
	AssertFO( growBytes > 0 );
	// Exponential enlargement, newBufferSize >= requiredSize
	int newBufferSize = max( bufferSize + bufferSize / 2, CeilTo( requiredSize, growBytes ) );
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

} // namespace NeoML