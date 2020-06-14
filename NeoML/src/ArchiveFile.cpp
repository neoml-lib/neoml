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

#include <NeoML/NeoMLDefs.h>
#include <NeoML/ArchiveFile.h>

#if FINE_PLATFORM( FINE_WINDOWS )
#include <io.h>
#elif FINE_PLATFORM( FINE_LINUX ) || FINE_PLATFORM( FINE_DARWIN ) || FINE_PLATFORM( FINE_IOS )
#include <cstdio>
#include <unistd.h>
#elif FINE_PLATFORM( FINE_ANDROID )
#include <jni.h>
#include <android/asset_manager.h>
#include <android/asset_manager_jni.h>
#else
#error Unknown platform
#endif

namespace NeoML {

static inline void throwFileException( int errorCode, const CString& fileName )
{
#ifdef NEOML_USE_FINEOBJ
	ThrowFileException( errorCode, fileName.CreateUnicodeString( CP_UTF8 ) );
#else
	ThrowFileException( errorCode, fileName );
#endif
}

#if FINE_PLATFORM( FINE_WINDOWS ) || FINE_PLATFORM( FINE_LINUX ) || FINE_PLATFORM( FINE_DARWIN ) || FINE_PLATFORM( FINE_IOS )

// Checks a condition and generates an exception if it is not fulfilled
// The _doserrno error code is used
static inline void checkArchiveFileError( bool condition, const CString& fileName )
{
	if( !condition ) {
#if FINE_PLATFORM( FINE_LINUX ) || FINE_PLATFORM( FINE_DARWIN ) || FINE_PLATFORM( FINE_IOS )
		throwFileException( errno, fileName );
#elif FINE_PLATFORM( FINE_WINDOWS )
		throwFileException( _doserrno, fileName );
#else
		#error Unknown platform
#endif
	}
}

//------------------------------------------------------------------------------------------------------------

CArchiveFile::CArchiveFile( const char* fileName, CArchive::TDirection direction, void* ) : 
	file( 0 )
{
	Open( fileName, direction );
}

void CArchiveFile::Open( const char* _fileName, CArchive::TDirection direction, void* )
{
	NeoAssert( !IsOpen() );
	try {
		fileName = _fileName;

		char mode[4]; // file opening parameters for _wfopen_s
		int nMode = 0; // the number of characters in mode

		if( direction == CArchive::SD_Loading ) {
			mode[nMode++] = 'r';
		} else if( direction == CArchive::SD_Storing ) {
			mode[nMode++] = 'w';
		} else {
			NeoAssert( false );
		}

		// Always open a file as binary
		mode[nMode++] = 'b';
		mode[nMode] = '\0';

#if FINE_PLATFORM( FINE_WINDOWS )
		checkArchiveFileError( fopen_s( reinterpret_cast<FILE**>( &file ), fileName, mode ) == 0, _fileName );
#elif FINE_PLATFORM( FINE_LINUX ) || FINE_PLATFORM( FINE_DARWIN ) || FINE_PLATFORM( FINE_IOS )
		file = fopen( fileName, mode );
		checkArchiveFileError( file != nullptr, _fileName );
#else
	#error Unknown platform
#endif
	} catch( ... ) {
		file = 0;
		fileName = CString();
		throw;
	}
}

int CArchiveFile::Read( void* buffer, int bytesCount )
{
	NeoAssert( IsOpen() );
	int bytesRead = static_cast<int>( fread( buffer, 1, bytesCount, reinterpret_cast<FILE*>( file ) ) );
	checkArchiveFileError( bytesRead != 0 || !feof( reinterpret_cast<FILE*>( file ) ), fileName );
	return bytesRead;
}

void CArchiveFile::Write( const void* buffer, int bytesCount )
{
	NeoAssert( IsOpen() );
	int bytesWritten = static_cast<int>( fwrite( buffer, 1, bytesCount, reinterpret_cast<FILE*>( file ) ) );
	checkArchiveFileError( bytesWritten == bytesCount, fileName );
}

__int64 CArchiveFile::GetPosition() const
{
	NeoAssert( IsOpen() );
#if FINE_PLATFORM( FINE_WINDOWS )
	__int64 position = _ftelli64( reinterpret_cast<FILE*>( file ) );
#elif FINE_PLATFORM( FINE_LINUX ) || FINE_PLATFORM( FINE_DARWIN ) || FINE_PLATFORM( FINE_IOS )
	static_assert( sizeof( off_t ) == sizeof( __int64 ),
		"sizeof(off_t) != sizeof(__int64)! Use _FILE_OFFSET_BITS=64 in compiler settings for 32-bit targets!" );
	__int64 position = ftello( static_cast<FILE*>( file ) );
#else
	#error Unknown platform
#endif
	checkArchiveFileError( position != -1, fileName );
	return position;
}

__int64 CArchiveFile::Seek( __int64 offset, TSeekPosition from )
{
	NeoAssert( IsOpen() );
#if FINE_PLATFORM( FINE_WINDOWS )
	checkArchiveFileError( _fseeki64( reinterpret_cast<FILE*>( file ), offset, from ) == 0, fileName );
	return _ftelli64( reinterpret_cast<FILE*>( file ) );
#elif FINE_PLATFORM( FINE_LINUX ) || FINE_PLATFORM( FINE_DARWIN ) || FINE_PLATFORM( FINE_IOS )
	checkArchiveFileError( fseeko( static_cast<FILE*>( file ), offset, from ) == 0, fileName );
	return ftello( static_cast<FILE*>( file) );
#else
	#error Unknown platform
#endif
}

void CArchiveFile::SetLength( __int64 newLength )
{
	NeoAssert( IsOpen() );
	// For the correct length flush the buffer to disk beforehand
	checkArchiveFileError( fflush( reinterpret_cast<FILE*>( file ) ) == 0, fileName );
#if FINE_PLATFORM( FINE_WINDOWS )
	checkArchiveFileError( _chsize_s( _fileno( reinterpret_cast<FILE*>( file ) ), newLength ) == 0, fileName );
#elif FINE_PLATFORM( FINE_LINUX ) || FINE_PLATFORM( FINE_DARWIN ) || FINE_PLATFORM( FINE_IOS )
	checkArchiveFileError( ftruncate( fileno( static_cast<FILE*>( file ) ), newLength ) == 0, fileName );
#else
	#error Unknown platform
#endif
}

__int64 CArchiveFile::GetLength() const
{
	NeoAssert( IsOpen() );
	auto f = static_cast<FILE*>( file );
#if FINE_PLATFORM( FINE_WINDOWS )
	// When experimenting we found that fflush + _filelengthi64 is faster than
	// a similar implementation of CStdioFile::GetLength() with ftell + fseek + ftell + fseek
	checkArchiveFileError( fflush( f ) == 0, fileName );
	__int64 length = _filelengthi64( _fileno( reinterpret_cast<FILE*>( file ) ) );
#elif FINE_PLATFORM( FINE_LINUX ) || FINE_PLATFORM( FINE_DARWIN ) || FINE_PLATFORM( FINE_IOS )
	checkArchiveFileError( fseeko( f, 0, SEEK_CUR ) == 0, fileName );
	off_t curPos = ftello( f );
	checkArchiveFileError( fseeko( f, 0, SEEK_END ) == 0, fileName );
	off_t length = ftello( f );
	checkArchiveFileError( fseeko( f, curPos, SEEK_SET ) == 0, fileName );
#else
	#error Unknown platform
#endif
	checkArchiveFileError( length != -1, fileName );
	return length;
}

void CArchiveFile::Abort()
{
	if( IsOpen() ) {
		fclose( reinterpret_cast<FILE*>( file ) );
		file = 0;
		fileName = CString();
	}
}

void CArchiveFile::Flush()
{
	NeoAssert( IsOpen() );
	checkArchiveFileError( fflush( reinterpret_cast<FILE*>( file ) ) == 0, fileName );
}

void CArchiveFile::Close()
{
	if( IsOpen() ) {
		// In case of error consider the file closed, so clear the filename and thread
		const CString fileNameTmp = fileName;
		fileName = CString();
		FILE* fileTmp = reinterpret_cast<FILE*>( file );
		file = 0;
		checkArchiveFileError( fclose( fileTmp ) == 0, fileNameTmp );
	}
}

bool CArchiveFile::IsEndOfFile() const
{
	NeoAssert( IsOpen() );
	return feof( reinterpret_cast<FILE*>( file ) ) != 0;
}

#elif FINE_PLATFORM( FINE_ANDROID )

CArchiveFile::CArchiveFile( const char* fileName, CArchive::TDirection direction, void* platformEnv ) :
	file( 0 )
{
	Open( fileName, direction, platformEnv );
}

void CArchiveFile::Open( const char* _fileName, CArchive::TDirection direction, void* platformEnv )
{
	NeoAssert( !IsOpen() );
	NeoAssert( direction == CArchive::SD_Loading );
	NeoAssert( platformEnv != nullptr );
	auto assetManager = static_cast<AAssetManager*>( platformEnv );
	file = AAssetManager_open( assetManager, _fileName, AASSET_MODE_RANDOM );
	if( file == nullptr ) {
		throwFileException( ENOENT, _fileName );
	}
	fileName = _fileName;
}

int CArchiveFile::Read( void* buffer, int bytesCount )
{
	NeoAssert( IsOpen() );
	int bytesRead = AAsset_read( static_cast<AAsset*>( file ), buffer, bytesCount );
	if( bytesRead <= 0 ) {
		throwFileException( 0, fileName );
	}
	return bytesRead;
}

void CArchiveFile::Write( const void*, int )
{
	NeoAssert( false );
}

__int64 CArchiveFile::GetPosition() const
{
	NeoAssert( IsOpen() );
	auto f = static_cast<AAsset*>( file );
	__int64 position = AAsset_getLength64( f ) - AAsset_getRemainingLength64( f );
	return position;
}

__int64 CArchiveFile::Seek( __int64 offset, TSeekPosition from )
{
	NeoAssert( IsOpen() );
	__int64 result = AAsset_seek64( static_cast<AAsset*>( file ), offset, from );
	if( result == -1 ) {
		throwFileException( EINVAL, fileName );
	}
	return result;
}

void CArchiveFile::SetLength( __int64 )
{
	NeoAssert( false );
}

__int64 CArchiveFile::GetLength() const
{
	NeoAssert( IsOpen() );
	return AAsset_getLength64( static_cast<AAsset*>( file ) );
}

void CArchiveFile::Abort()
{
	if( IsOpen() ) {
		AAsset_close( static_cast<AAsset*>( file ) );
		file = nullptr;
		fileName = CString();
	}
}

void CArchiveFile::Flush()
{
	NeoAssert( false );
}

void CArchiveFile::Close()
{
	if( IsOpen() ) {
		AAsset_close( static_cast<AAsset*>( file ) );
		file = nullptr;
		fileName = CString();
	}
}

bool CArchiveFile::IsEndOfFile() const
{
	NeoAssert( IsOpen() );
	return AAsset_getRemainingLength64( static_cast<AAsset*>( file ) ) == 0;
}

#else
	#error Unknown platform
#endif

void CArchiveFile::ReadRecord( void* buff, int size )
{
	if( Read( buff, size ) != size ) {
		#if FINE_PLATFORM( FINE_WINDOWS )
			ThrowFileException( ERROR_HANDLE_EOF, GetFileName() );
		#elif FINE_PLATFORM( FINE_ANDROID ) || FINE_PLATFORM( FINE_IOS ) || FINE_PLATFORM( FINE_LINUX ) || FINE_PLATFORM( FINE_DARWIN )
			ThrowFileException( EOVERFLOW, GetFileName() );
		#else
			#error "Platform isn't supported!"
		#endif
	}
}

unsigned char CArchiveFile::ReadByte()
{
	unsigned char ret = 0;
	ReadRecord( &ret, sizeof( ret ) );
	return ret;
}

} // namespace NeoML
