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

#include "ErrorsFOL.h"

namespace FObj {

// The base class for binary files
class CBaseFile {
public:
	// The starting position for the Seek method
	enum TSeekPosition {
		begin = 0,
		current = 1,
		end = 2
	};

	CBaseFile() {}
	virtual ~CBaseFile() {}
	// Reads the name of the file
	virtual const char* GetFileName() const = 0;
	// Reads bytesCount bytes from the file, returns the number of bytes actually read
	virtual int Read( void*, int bytesCount ) = 0;
	// Writes bytesCount bytes into the file
	virtual void Write( const void*, int bytesCount ) = 0;
	// Gets the current position in the file
	virtual __int64 GetPosition() const = 0;
	int GetPosition32() const
	{
		__int64 position = GetPosition();
		AssertFO( 0 <= position && position <= INT_MAX );
		return static_cast<int>( position );
	}
	// Changes the current position in the file
	virtual __int64 Seek( __int64 offset, TSeekPosition from ) = 0;
	int Seek32( int offset, TSeekPosition from )
	{
		__int64 position = Seek( offset, from );
		AssertFO( 0 <= position && position <= INT_MAX );
		return static_cast<int>( position );
	}
	void SeekToBegin() { Seek( 0, begin ); }
	void SeekToEnd() { Seek( 0, end ); }
	// Sets the file length
	virtual void SetLength( __int64 newLength ) = 0;
	void SetLength32( int newLength ) { SetLength( newLength ); }
	// Retrieves the file length
	virtual __int64 GetLength() const = 0;
	int GetLength32() const
	{
		__int64 length = GetLength();
		AssertFO( 0 <= length && length <= INT_MAX );
		return static_cast<int>( length );
	}
	// Closes the file ignoring all errors (no exceptions thrown)
	// This method may be called for a file that hasn't been opened
	virtual void Abort() = 0;
	// Writes the file contents on disk
	virtual void Flush() = 0;
	// Closes the file
	virtual void Close() = 0;
	// Checks if the current position is at the end of the file
	virtual bool IsEndOfFile() const { return GetPosition() == GetLength(); }

private:
	CBaseFile( const CBaseFile& );
	CBaseFile& operator=( const CBaseFile& );
};

} // namespace FObj
