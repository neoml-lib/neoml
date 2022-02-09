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

#include <AllocFOL.h>
#include <ErrorsFOL.h>

namespace NeoML {

// The class for working with in-memory files.
template<class Allocator = CurrentMemoryManager>
class CMemoryFileEx : public CBaseFile {
public:
	explicit CMemoryFileEx( int growBytes = 1024 );
	virtual ~CMemoryFileEx();

	bool IsOpen() const { return isOpen; }

	// CBaseFile methods:
#ifdef FINEOBJ_VERSION
	virtual CUnicodeString GetFileName() const { return L"Memory file."; }
#else
	virtual const char* GetFileName() const { return "Memory file."; }
#endif
	virtual int Read( void*, int bytesCount );
	virtual void Write( const void*, int bytesCount );
	virtual void Close();
	virtual __int64 GetPosition() const;
	virtual __int64 Seek( __int64 offset, TSeekPosition from );
	virtual void SetLength( __int64 newLength );
	virtual __int64 GetLength() const;
	virtual void Abort();
	virtual void Flush();

protected:
	virtual void FreeBuffer( BYTE* );
	virtual BYTE* GrowBuffer( BYTE*, int oldSize, int newSize );

private:
	BYTE* buffer; // file buffer
	int bufferSize; // current buffer size
	int fileLength; // file size
	int growBytes; // minimum increment of buffer size
	int currentPosition; // file pointer current offset
	bool isOpen; // whether file is open

	void setBufferSize( int requiredSize );
	void throwBadSeekException();
};

inline CMemoryFileEx::CMemoryFile( int _growBytes ) :
	buffer( 0 ),
	bufferSize( 0 ),
	fileLength( 0 ),
	growBytes( _growBytes ),
	currentPosition( 0 ),
	isOpen( true )
{
	NeoAssert( growBytes >= 0 );
}

inline CMemoryFileEx::~CMemoryFile()
{
	Close();
}

inline int CMemoryFileEx::Read( void* ptr, int bytesCount )
{
	if( bytesCount == 0 ) {
		return 0;
	}
	NeoAssert( ptr != 0 );
	NeoAssert( bytesCount > 0 );

	int size = min( fileLength - currentPosition, bytesCount );
	if( size <= 0 ) {
		return 0;
	}
	::memcpy( ptr, buffer + currentPosition, size );
	currentPosition += size;
	return size;
}

inline void CMemoryFileEx::Write( const void* ptr, int bytesCount )
{
	if( bytesCount == 0 ) {
		return;
	}

	NeoAssert( ptr != 0 );
	NeoAssert( bytesCount > 0 );

	int newPosition = currentPosition + bytesCount;
	if( newPosition > bufferSize ) {
		setBufferSize( newPosition );
	}
	::memcpy( buffer + currentPosition, ptr, bytesCount );
	currentPosition = newPosition;
	fileLength = max( fileLength, currentPosition );
}

inline void CMemoryFileEx::Close()
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

inline __int64 CMemoryFileEx::GetPosition() const
{
	return currentPosition;
}

inline __int64 CMemoryFileEx::Seek( __int64 offset, TSeekPosition from )
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
			NeoAssert( false );
	}
	if( 0 <= newPosition && newPosition <= INT_MAX ) {
		currentPosition = to<int>( newPosition );
	} else {
		currentPosition = 0;
		throwBadSeekException();
	}
	return currentPosition;
}

inline void CMemoryFileEx::SetLength( __int64 newLength )
{
	NeoAssert( 0 <= newLength && newLength <= INT_MAX );
	int length32 = to<int>( newLength );
	if( bufferSize < length32 ) {
		setBufferSize( length32 );
	}
	if( length32 < currentPosition ) {
		currentPosition = length32;
	}
	fileLength = length32;
}

inline __int64 CMemoryFileEx::GetLength() const
{
	return fileLength;
}

inline void CMemoryFileEx::Abort()
{
	Close();
}

inline void CMemoryFileEx::Flush()
{
}

inline void CMemoryFileEx::FreeBuffer( BYTE* ptr )
{
	CurrentMemoryManager::Free( ptr );
}

inline BYTE* CMemoryFileEx::GrowBuffer( BYTE* oldBuffer, int oldSize, int newSize )
{
	NeoAssert( newSize > oldSize );
	BYTE* newBuffer = static_cast<BYTE*>( ALLOCATE_MEMORY( CurrentMemoryManager, newSize * sizeof( BYTE ) ) );
	if( oldSize > 0 ) {
		::memcpy( newBuffer, oldBuffer, oldSize );
	}
	if( oldBuffer != 0 ) {
		FreeBuffer( oldBuffer );
	}
	return newBuffer;
}

inline void CMemoryFileEx::setBufferSize( int requiredSize )
{
	NeoAssert( growBytes > 0 );
	// Exponential enlargement, newBufferSize >= requiredSize
	int newBufferSize = max( bufferSize + bufferSize / 2, CeilTo( requiredSize, growBytes ) );
	buffer = GrowBuffer( buffer, bufferSize, newBufferSize );
	NeoAssert( buffer != 0 );
	bufferSize = newBufferSize;
}

inline void CMemoryFileEx::throwBadSeekException()
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