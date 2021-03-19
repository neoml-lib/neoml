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

#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl_bind.h>

namespace py = pybind11;

class CPyMemoryFile : public CBaseFile {
public:
	CPyMemoryFile();
	CPyMemoryFile( const py::array& buffer );
	virtual ~CPyMemoryFile();

	bool IsOpen() const { return state != S_Closed; }
	const py::array& GetBuffer() const;

	virtual const char* GetFileName() const { return "Memory file."; }
	virtual int Read( void*, int bytesCount );
	virtual void Write( const void*, int bytesCount );
	virtual void Close();
	virtual long long GetPosition() const;
	virtual long long Seek( long long offset, TSeekPosition from );
	virtual void SetLength( long long newLength );
	virtual long long GetLength() const;
	virtual void Abort();
	virtual void Flush();

private:
	static const int growBytes = 1024;

	enum TState {
		S_Read = 0,
		S_Write,
		S_Closed
	};

	py::array buffer;
	int bufferSize;
	int fileLength;
	int currentPosition;
	TState state;

	void setBufferSize( int requiredSize );
};
