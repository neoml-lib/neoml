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

#include <common.h>
#pragma hdrstop

#include "PyMemoryFile.h"

CPyMemoryFile::CPyMemoryFile() :
	buffer( py::array( py::dtype("byte"), {}, {} ) ),
	bufferSize( 0 ),
	fileLength( 0 ),
	currentPosition( 0 ),
	state( S_Write )
{
	NeoAssert( growBytes >= 0 );
}

CPyMemoryFile::CPyMemoryFile( py::array _buffer ) :
	buffer( _buffer ),
	bufferSize( static_cast<int>( _buffer.size() ) ),
	fileLength( static_cast<int>( _buffer.size() ) ),
	currentPosition( 0 ),
	state( S_Read )
{
	NeoAssert( growBytes >= 0 );
}

CPyMemoryFile::~CPyMemoryFile()
{
	Close();
}

py::array CPyMemoryFile::GetBuffer() const
{
	NeoAssert( !IsOpen() );
	return buffer;
}

int CPyMemoryFile::Read( void* ptr, int bytesCount )
{
	NeoAssert( state == S_Read );
	if( bytesCount == 0 ) {
		return 0;
	}
	NeoAssert( ptr != 0 );
	NeoAssert( bytesCount > 0 );

	int size = min( fileLength - currentPosition, bytesCount );
	if( size <= 0 ) {
		return 0;
	}
	::memcpy( ptr, (char*)buffer.mutable_data() + currentPosition, size );
	currentPosition += size;
	return size;
}

void CPyMemoryFile::Write( const void* ptr, int bytesCount )
{
	NeoAssert( state == S_Write );

	if( bytesCount == 0 ) {
		return;
	}

	NeoAssert( ptr != 0 );
	NeoAssert( bytesCount > 0 );

	int newPosition = currentPosition + bytesCount;
	if( newPosition > bufferSize ) {
		setBufferSize( newPosition );
	}
	::memcpy( (char*)buffer.mutable_data() + currentPosition, ptr, bytesCount );
	currentPosition = newPosition;
	fileLength = max( fileLength, currentPosition );
}

void CPyMemoryFile::Close()
{
	if( !IsOpen() ) {
		return;
	}

	buffer.resize( {fileLength} );
	bufferSize = 0;
	fileLength = 0;
	currentPosition = 0;
	state = S_Closed;
}

long long CPyMemoryFile::GetPosition() const
{
	return currentPosition;
}

long long CPyMemoryFile::Seek( long long offset, TSeekPosition from )
{
	NeoAssert( IsOpen() );

	long long newPosition = currentPosition;
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
	}
	return currentPosition;
}

void CPyMemoryFile::SetLength( long long newLength )
{
	NeoAssert( state == S_Write );
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

long long CPyMemoryFile::GetLength() const
{
	return fileLength;
}

void CPyMemoryFile::Abort()
{
	Close();
}

void CPyMemoryFile::Flush()
{
}

void CPyMemoryFile::setBufferSize( int requiredSize )
{
	NeoAssert( growBytes > 0 );
	int newBufferSize = max( bufferSize + bufferSize / 2, CeilTo( requiredSize, growBytes ) );
	buffer.resize( {newBufferSize} );
	bufferSize = newBufferSize;
}
