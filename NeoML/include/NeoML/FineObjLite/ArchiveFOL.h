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

#include "BaseFileFOL.h"

namespace FObj {

// Archive over a binary file
// Used for serialization
class CArchive {
public:
	enum TDirection {
		SD_Undefined,
		SD_Loading,
		SD_Storing,
		load = SD_Loading,
		store = SD_Storing
	};

	explicit CArchive();
	CArchive( CBaseFile* baseFile, TDirection direction );
	virtual ~CArchive();

	const char* Name() const { return name; }

	void Open( CBaseFile* baseFile, TDirection direction );
	void Close();
	bool IsOpen() const { return file != 0; }

	void Read( void* ptr, int size );
	void Write( const void* ptr, int size );
	void Flush();
	// Skips the size bytes
	void Skip( int size );
	// Copies the data into another archive
	void CopyTo( CArchive& dest, __int64 size );
	void Abort();

	bool IsLoading() const { return direction == SD_Loading; }
	bool IsStoring() const { return direction == SD_Storing; }
	bool IsEndOfArchive() const;
	// Gets the current position in archive
	// Note that it may not be the same as the position in file ( GetFile()->GetPosition() )
	// because the archive may be reading/writing with offset from the file beginning
	__int64 GetPosition() const;
	int GetPosition32() const;
	// Navigate through file
	__int64 Seek( __int64 offset, CBaseFile::TSeekPosition from );
	int Seek32( int offset, CBaseFile::TSeekPosition from );
	// Gets the current archive length
	// Note that it may not be the same as file length because some of the data may not have been written into the file yet
	__int64 GetLength() const;
	int GetLength32() const;

	// Read and write standard data types
	friend CArchive& operator <<( CArchive&, const CString& string );
	friend CArchive& operator <<( CArchive&, char variable );
	friend CArchive& operator <<( CArchive&, signed char variable );
	friend CArchive& operator <<( CArchive&, wchar_t variable );
	friend CArchive& operator <<( CArchive&, bool variable );
	friend CArchive& operator <<( CArchive&, short variable );
	friend CArchive& operator <<( CArchive&, int variable );
	friend CArchive& operator <<( CArchive&, __int64 variable );
	friend CArchive& operator <<( CArchive&, float variable );
	friend CArchive& operator <<( CArchive&, double variable );
	friend CArchive& operator <<( CArchive&, unsigned char variable );
	friend CArchive& operator <<( CArchive&, unsigned short variable );
	friend CArchive& operator <<( CArchive&, unsigned int variable );
	friend CArchive& operator <<( CArchive&, unsigned __int64 variable );
	friend CArchive& operator >>( CArchive&, CString& string );
	friend CArchive& operator >>( CArchive&, char& variable );
	friend CArchive& operator >>( CArchive&, signed char& variable );
	friend CArchive& operator >>( CArchive&, wchar_t& variable );
	friend CArchive& operator >>( CArchive&, bool& variable );
	friend CArchive& operator >>( CArchive&, short& variable );
	friend CArchive& operator >>( CArchive&, int& variable );
	friend CArchive& operator >>( CArchive&, __int64& variable );
	friend CArchive& operator >>( CArchive&, float& variable );
	friend CArchive& operator >>( CArchive&, double& variable );
	friend CArchive& operator >>( CArchive&, unsigned char& variable );
	friend CArchive& operator >>( CArchive&, unsigned short& variable );
	friend CArchive& operator >>( CArchive&, unsigned int& variable );
	friend CArchive& operator >>( CArchive&, unsigned __int64& variable );

	// Read and write small integers
	// Numbers 0 to 254 will be serialized in 1 byte, other numbers in 5 bytes
	void SerializeSmallValue( unsigned int& );
	void SerializeSmallValue( int& );
	int ReadSmallValue();
	void WriteSmallValue( int );

	// Read and write versions
	int SerializeVersion( int currentVersion );
	int SerializeVersion( int currentVersion, int minSupportedVersion );

	template<class T>
	void Serialize( T& variable );
	template<class T>
	void SerializeEnum( T& variable );

private:
	const int DefaultArchiveBufferSize = 4096;

	CBaseFile* file;
	CString name;
	TDirection direction;
	char buffer[4096];
	int bufferSize;
	__int64 beginOfArchive;
	__int64 filePosition;
	__int64 fileLength;
	int currentPosition;
	int leftInBuffer;
	bool isActualizedFileParameters;

	template<class T>
	void writeSimpleType( T object );
	template<class T>
	void readSimpleType( T& object );

	void actualizeFileParameters();
	void readOverBuffer( void* ptr, int size );
	void writeOverBuffer( const void* ptr, int size );
	int peek( void* resultPtr, int size ) const;
	void seekWhenLoading( __int64 newArchivePosition );
	void seekWhenStoring( __int64 newArchivePosition );
	void throwEofException();
};

inline CArchive::CArchive( CBaseFile* _file, CArchive::TDirection _direction ) :
	file( 0 ),
	direction( SD_Undefined ),
	bufferSize( DefaultArchiveBufferSize ),
	beginOfArchive( 0 ),
	filePosition( 0 ),
	fileLength( 0 ),
	currentPosition( 0 ),
	leftInBuffer( 0 ),
	isActualizedFileParameters( false )
{
	Open( _file, _direction );
}

inline CArchive::CArchive() :
	file( 0 ),
	direction( SD_Undefined ),
	bufferSize( DefaultArchiveBufferSize ),
	beginOfArchive( 0 ),
	filePosition( 0 ),
	fileLength( 0 ),
	currentPosition( 0 ),
	leftInBuffer( 0 ),
	isActualizedFileParameters( false )
{
}

inline void CArchive::Open( CBaseFile* _file, CArchive::TDirection _direction )
{
	AssertFO( file == 0 );
	AssertFO( _file != 0 );

	file = _file;
	direction = _direction;
	name = file->GetFileName();
	beginOfArchive = 0;
	filePosition = 0;
	fileLength = 0;
	isActualizedFileParameters = false;
	currentPosition = 0;
	leftInBuffer = 0;
}

inline CArchive::~CArchive()
{
	Close();
}

inline void CArchive::Close()
{
	if( file == 0 ) {
		return;
	}

	Flush();
	file = 0;
	name = CString();
	direction = SD_Undefined;
}

inline void CArchive::Read( void* ptr, int size )
{
	AssertFO( file != 0 );
	AssertFO( size >= 0 );
	AssertFO( IsLoading() );

	if( size == 0 ) {
		return;
	}
	if( size <= leftInBuffer ) {
		::memcpy( ptr, buffer + currentPosition, size );
		currentPosition += size;
		leftInBuffer -= size;
		return;
	}

	readOverBuffer( ptr, size );
}

inline void CArchive::Write( const void* ptr, int size )
{
	AssertFO( file != 0 );
	AssertFO( size >= 0 );
	AssertFO( IsStoring() );

	if( size == 0 ) {
		return;
	}
	if( size + currentPosition < bufferSize ) {
		::memcpy( buffer + currentPosition, ptr, size );
		currentPosition += size;
		leftInBuffer = max( leftInBuffer - size, 0 );
		return;
	}

	writeOverBuffer( ptr, size );
}

inline void CArchive::Flush()
{
	AssertFO( file != 0 );

	if( IsLoading() ) {
		if( leftInBuffer > 0 ) {
			file->Seek( - static_cast<__int64>( leftInBuffer ), CBaseFile::current );
			filePosition -= static_cast<__int64>( leftInBuffer );
		}
	} else { 
		if( currentPosition + leftInBuffer > 0 ) {
			file->Write( buffer, currentPosition + leftInBuffer );
			fileLength = max( fileLength, filePosition + currentPosition + leftInBuffer );
			if( leftInBuffer != 0 ) {
				file->Seek( - static_cast<__int64>( leftInBuffer ), CBaseFile::current );
			}
			filePosition += currentPosition;
		}
	}
	currentPosition = 0;
	leftInBuffer = 0;
}

inline void CArchive::Skip( int size )
{
	AssertFO( file != 0 );
	AssertFO( size >= 0 );

	if( size == 0 ) {
		return;
	}

	if( IsLoading() ) {
		if( size < leftInBuffer ) {
			currentPosition += size;
			leftInBuffer -= size;
		} else {
			if( beginOfArchive + GetPosition() + static_cast<__int64>( size ) > file->GetLength() ) {
				throwEofException();
			}
			file->Seek( static_cast<__int64>( size - leftInBuffer ), CBaseFile::current );
			filePosition += size - leftInBuffer;
			currentPosition = 0;
			leftInBuffer = 0;
		}
	} else if( IsStoring() ) {
		if( size < bufferSize - currentPosition ) {
			currentPosition += size;
			leftInBuffer = max( leftInBuffer - size, 0 );
		} else {
			Flush();
			file->Seek( static_cast<__int64>( size ), CBaseFile::current );

			filePosition += size;
			if( filePosition > fileLength ) {
				fileLength = filePosition;
				if( !isActualizedFileParameters ) {
					actualizeFileParameters();
				}
				file->SetLength( fileLength );
			}
		}
	} else {
		AssertFO( false );
	}
}

inline void CArchive::CopyTo( CArchive& dest, __int64 size )
{
	AssertFO( IsOpen() );
	AssertFO( IsLoading() );
	AssertFO( dest.IsOpen() );
	AssertFO( dest.IsStoring() );
	AssertFO( size >= 0 );

	while( size > 0 ) {
		if( leftInBuffer == 0 ) {
			currentPosition = 0;

			const int readBufferSize = ( bufferSize > 0 ) ? bufferSize
				: static_cast<int>( min( size, static_cast<__int64>( DefaultArchiveBufferSize ) ) );
			leftInBuffer = file->Read( buffer, readBufferSize );
			filePosition += leftInBuffer;
			if( leftInBuffer < static_cast<int>( min( size, static_cast<__int64>( readBufferSize ) ) ) ) {
				leftInBuffer = 0;
				throwEofException();
			}
			if( size >= readBufferSize ) {
				dest.Flush();
			}
		}

		int byteToCopy = static_cast<int>( min( size, static_cast<__int64>( leftInBuffer ) ) );
		dest.Write( buffer + currentPosition, byteToCopy );

		leftInBuffer -= byteToCopy;
		currentPosition = leftInBuffer == 0 ? 0 : currentPosition + byteToCopy;
		size -= byteToCopy;
	}
}

inline void CArchive::Abort()
{
	file = 0;
	name = CString();
	direction = SD_Undefined;
}

inline bool CArchive::IsEndOfArchive() const
{
	AssertFO( IsLoading() && file != 0 );
	return leftInBuffer == 0 && file->IsEndOfFile();
}

inline int CArchive::GetPosition32() const
{
	__int64 result = GetPosition();
	AssertFO( 0 <= result && result <= INT_MAX );
	return static_cast<int>( result );
}

inline __int64 CArchive::GetPosition() const
{
	AssertFO( file != 0 );
	if( IsLoading() ) {
		return filePosition - beginOfArchive - static_cast<__int64>( leftInBuffer );
	} else {
		return filePosition - beginOfArchive + static_cast<__int64>( currentPosition );
	}
}

inline int CArchive::Seek32( int offset, CBaseFile::TSeekPosition from )
{
	__int64 result = Seek( to<__int64>( offset ), from );
	assert( result <= INT_MAX );
	return to<int>( result );
}

inline __int64 CArchive::Seek( __int64 offset, CBaseFile::TSeekPosition from )
{
	assert( file != 0 );
	if( !isActualizedFileParameters ) {
		actualizeFileParameters();
	}
	__int64 newArchivePosition = 0;

	switch( from ) {
		case CBaseFile::current:
			newArchivePosition = GetPosition() + offset;
			break;
		case CBaseFile::begin:
			newArchivePosition = offset;
			break;
		case CBaseFile::end:
			newArchivePosition = GetLength() + offset;
			break;
		default:
			assert( false );
	}
	if( newArchivePosition < 0 || newArchivePosition > GetLength() ) {
		throwEofException();
	}

	if( IsLoading() ) {
		seekWhenLoading( newArchivePosition );
	} else { // IsStoring
		seekWhenStoring( newArchivePosition );
	}
	return GetPosition();
}

inline int CArchive::GetLength32() const
{
	__int64 result = GetLength();
	AssertFO( result <= INT_MAX );
	return static_cast<int>( result );
}

inline __int64 CArchive::GetLength() const
{
	AssertFO( file != 0 );
	if( !isActualizedFileParameters ) {
		const_cast<CArchive*>( this )->actualizeFileParameters();
	}
	return max( fileLength - beginOfArchive, GetPosition() + leftInBuffer );
}

inline CArchive& operator<<( CArchive& stream, const CString& string )
{
	stream.WriteSmallValue( static_cast<int>( string.length() ) );
	stream.Write( string.data(), static_cast<int>( string.length() ) );
	return stream;
}

inline CArchive& operator<<( CArchive& archive, char variable )
{
	archive.writeSimpleType( variable );
	return archive;
}

inline CArchive& operator<<( CArchive& archive, signed char variable )
{
	archive.writeSimpleType( variable );
	return archive;
}

inline CArchive& operator<<( CArchive& archive, wchar_t variable )
{
	archive.writeSimpleType( variable );
	return archive;
}

inline CArchive& operator<<( CArchive& archive, bool variable )
{
	char byte = static_cast<char>( variable );
	AssertFO( byte == 0 || byte == 1 );
	archive.writeSimpleType( byte );
	return archive;
}

inline CArchive& operator<<( CArchive& archive, short variable )
{
	archive.writeSimpleType( variable );
	return archive;
}

inline CArchive& operator<<( CArchive& archive, int variable )
{
	archive.writeSimpleType( variable );
	return archive;
}

inline CArchive& operator<<( CArchive& archive, __int64 variable )
{
	archive.writeSimpleType( variable );
	return archive;
}

inline CArchive& operator<<( CArchive& archive, float variable )
{
	archive.writeSimpleType( variable );
	return archive;
}

inline CArchive& operator<<( CArchive& archive, double variable )
{
	archive.writeSimpleType( variable );
	return archive;
}

inline CArchive& operator<<( CArchive& archive, unsigned char variable )
{
	archive.writeSimpleType( variable );
	return archive;
}

inline CArchive& operator<<( CArchive& archive, unsigned short variable )
{
	archive.writeSimpleType( variable );
	return archive;
}

inline CArchive& operator<<( CArchive& archive, unsigned int variable )
{
	archive.writeSimpleType( variable );
	return archive;
}

inline CArchive& operator<<( CArchive& archive, unsigned __int64 variable )
{
	archive.writeSimpleType( variable );
	return archive;
}

inline CArchive& operator>>( CArchive& stream, CString& string )
{
	string.erase();
	int length = stream.ReadSmallValue();
	check( length >= 0, ERR_BAD_ARCHIVE, stream.Name() );
	if( length == 0 ) {
		return stream;
	}
	string.resize( length );
	char* ptr = const_cast<char*>( string.data() );
	stream.Read( ptr, length );
	return stream;
}

inline CArchive& operator>>( CArchive& archive, char& variable )
{
	archive.readSimpleType( variable );
	return archive;
}

inline CArchive& operator>>( CArchive& archive, signed char& variable )
{
	archive.readSimpleType( variable );
	return archive;
}

inline CArchive& operator>>( CArchive& archive, wchar_t& variable )
{
	archive.readSimpleType( variable );
	return archive;
}

inline CArchive& operator>>( CArchive& archive, bool& variable )
{
	char result;
	archive.readSimpleType( result );
	check( result == 0 || result == 1, ERR_BAD_ARCHIVE, archive.Name() );
	variable = result != 0;
	return archive;
}

inline CArchive& operator>>( CArchive& archive, short& variable )
{
	archive.readSimpleType( variable );
	return archive;
}

inline CArchive& operator>>( CArchive& archive, int& variable )
{
	archive.readSimpleType( variable );
	return archive;
}

inline CArchive& operator>>( CArchive& archive, __int64& variable )
{
	archive.readSimpleType( variable );
	return archive;
}

inline CArchive& operator>>( CArchive& archive, float& variable )
{
	archive.readSimpleType( variable );
	return archive;
}

inline CArchive& operator>>( CArchive& archive, double& variable )
{
	archive.readSimpleType( variable );
	return archive;
}

inline CArchive& operator>>( CArchive& archive, unsigned char& variable )
{
	archive.readSimpleType( variable );
	return archive;
}

inline CArchive& operator>>( CArchive& archive, unsigned short& variable )
{
	archive.readSimpleType( variable );
	return archive;
}

inline CArchive& operator>>( CArchive& archive, unsigned int& variable )
{
	archive.readSimpleType( variable );
	return archive;
}

inline CArchive& operator>>( CArchive& archive, unsigned __int64& variable )
{
	archive.readSimpleType( variable );
	return archive;
}

template<class T>
inline void CArchive::Serialize( T& variable )
{
	if( IsLoading() ) {
		*this >> variable;
	} else {
		*this << variable;
	}
}

template<class T>
inline void CArchive::SerializeEnum( T& variable )
{
	if( IsLoading() ) {
		variable = static_cast<T>( ReadSmallValue() );
	} else {
		WriteSmallValue( variable );
	}
}

template<class T>
inline void CArchive::writeSimpleType( T object )
{
	Write( &object, sizeof( T ) );
}

template<class T>
inline void CArchive::readSimpleType( T& object )
{
	Read( &object, sizeof( T ) );
}

inline void CArchive::SerializeSmallValue( unsigned int& value )
{
	if( IsLoading() ) {
		value = static_cast<unsigned int>( ReadSmallValue() );
	} else {
		WriteSmallValue( static_cast<int>( value ) );
	}
}

inline void CArchive::SerializeSmallValue( int& value )
{
	if( IsLoading() ) {
		value = ReadSmallValue();
	} else {
		WriteSmallValue( value );
	}
}

inline int CArchive::SerializeVersion( int currentVersion )
{
	if( IsStoring() ) {
		WriteSmallValue( currentVersion );
		return currentVersion;
	} else {
		int version = ReadSmallValue();
		if( version > currentVersion ) {
			check( false, ERR_BAD_ARCHIVE_VERSION, Name() );
		}
		return version;
	}
}

inline int CArchive::SerializeVersion( int currentVersion, int minSupportedVersion )
{
	if( IsStoring() ) {
		WriteSmallValue( currentVersion );
		return currentVersion;
	} else {
		int version = ReadSmallValue();
		if( version < minSupportedVersion || version > currentVersion ) {
			check( false, ERR_BAD_ARCHIVE_VERSION, Name() );
		}
		return version;
	}
}

inline void CArchive::readOverBuffer( void* ptr, int size )
{
	char* readPtr = reinterpret_cast<char*>( ptr );
	if( 0 < leftInBuffer ) {
		::memcpy( readPtr, buffer + currentPosition, leftInBuffer );
		readPtr += leftInBuffer;
		size -= leftInBuffer;
		leftInBuffer = 0;
	}
	currentPosition = 0;
	if( bufferSize <= size ) {
		int bytesFromFile = file->Read( readPtr, size );
		if( bytesFromFile != size ) {
			throwEofException();
		}
		filePosition += bytesFromFile;
	} else {
		leftInBuffer = file->Read( buffer, bufferSize );
		if( leftInBuffer < size ) {
			throwEofException();
		}
		filePosition += leftInBuffer;
		::memcpy( readPtr, buffer, size );
		currentPosition += size;
		leftInBuffer -= size;
	}
}

inline void CArchive::writeOverBuffer( const void* ptr, int size )
{
	const char* writePtr = reinterpret_cast<const char*>( ptr );
	if( currentPosition > 0 ) {
		int toBuffer = bufferSize - currentPosition;
		::memcpy( buffer + currentPosition, writePtr, toBuffer );
		currentPosition = bufferSize;
		leftInBuffer = 0;
		writePtr += toBuffer;
		size -= toBuffer;
		Flush();
	}
	if( bufferSize <= size ) {
		file->Write( writePtr, size );
		filePosition += size;
	} else {
		::memcpy( buffer, writePtr, size );
		currentPosition = size;
	}
	fileLength = max( fileLength, filePosition );
	leftInBuffer = 0;
}

inline void CArchive::actualizeFileParameters()
{
	AssertFO( !isActualizedFileParameters );
	beginOfArchive = file->GetPosition() - filePosition;
	AssertFO( beginOfArchive >= 0 );
	filePosition += beginOfArchive;
	fileLength = max( file->GetLength(), beginOfArchive + fileLength );
	isActualizedFileParameters = true;
}


inline void CArchive::seekWhenLoading( __int64 newArchivePosition )
{
	__int64 newCurrentPosition = static_cast<__int64>( currentPosition ) + newArchivePosition - GetPosition();
	if( newCurrentPosition >= 0 && newCurrentPosition <= static_cast<__int64>( currentPosition ) + leftInBuffer ) {
		leftInBuffer -= static_cast<int>( newCurrentPosition ) - currentPosition;
		currentPosition = static_cast<int>( newCurrentPosition );
	} else {
		file->Seek( beginOfArchive + newArchivePosition, CBaseFile::begin );
		filePosition = beginOfArchive + newArchivePosition;
		currentPosition = 0;
		leftInBuffer = 0;
	}
}

inline void CArchive::seekWhenStoring( __int64 newArchivePosition )
{
	__int64 newCurrentPosition = static_cast<__int64>( currentPosition ) + newArchivePosition - GetPosition();
	if( newCurrentPosition >= 0 && newCurrentPosition <= static_cast<__int64>( currentPosition ) + leftInBuffer ) {
		leftInBuffer -= static_cast<int>( newCurrentPosition ) - currentPosition;
		currentPosition = static_cast<int>( newCurrentPosition );
	} else {
		Flush();
		file->Seek( beginOfArchive + newArchivePosition, CBaseFile::begin );
		filePosition = beginOfArchive + newArchivePosition;
	}
}

inline int CArchive::peek( void* resultPtr, int size ) const
{
	AssertFO( IsLoading() );
	PresumeFO( resultPtr != 0 );
	PresumeFO( 0 <= size );
	
	char* writePtr = reinterpret_cast<char*>( resultPtr );
	
	int fromBufferSize = min( leftInBuffer, size );
	if( fromBufferSize > 0 ) {
		::memcpy( writePtr, buffer + currentPosition, fromBufferSize );
		writePtr += fromBufferSize;
	}
	
	int fromFileSize = size - fromBufferSize;
	if( fromFileSize > 0 ) {
		fromFileSize = file->Read( writePtr, fromFileSize );
		file->Seek( - static_cast<__int64>( fromFileSize ), CBaseFile::current );
	}
	return fromBufferSize + fromFileSize;
}

inline int CArchive::ReadSmallValue()
{
	unsigned char firstByte;
	*this >> firstByte;
	if( firstByte != UCHAR_MAX ) {
		return firstByte;
	}
	int ret;
	*this >> ret;
	return ret;
}

inline void CArchive::WriteSmallValue( int value )
{
	if( 0 <= value && value < UCHAR_MAX ) {
		*this << ( unsigned char )( DWORD )value;
	} else {
		*this << ( unsigned char )(UCHAR_MAX) << value;
	}
}

inline void CArchive::throwEofException()
{
#if FINE_PLATFORM( FINE_WINDOWS )
	ThrowFileException( ERROR_HANDLE_EOF, Name() );
#elif FINE_PLATFORM( FINE_ANDROID ) || FINE_PLATFORM( FINE_IOS ) || FINE_PLATFORM( FINE_LINUX ) || FINE_PLATFORM( FINE_DARWIN )
	ThrowFileException( EOVERFLOW, Name() );
#else
	#error "Platform is not supported!"
#endif
}

} // namespace FObj
