/* Copyright © 2017-2021 ABBYY Production LLC

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

#include "PyMathEngine.h"

py::array CreateArray( const CDnnBlob& blob );

CPtr<CDnnBlob> CreateBlob( IMathEngine& mathEngine, const py::array& data );

//------------------------------------------------------------------------------------------------------------

class CPyBlob {
public:
	CPyBlob( const CPyMathEngine& pyMathEngine, TBlobType type, int batchLength, int batchWidth, int listSize,
		int height, int width, int depth, int channels );
	CPyBlob( const CPyMathEngine& pyMathEngine, TBlobType blobType, int batchLength, int batchWidth, int listSize,
		int height, int width, int depth, int channels, py::buffer data, bool copy );
	CPyBlob( CPyMathEngineOwner& _mathEngineOwner, CDnnBlob* _blob );

	py::object GetMathEngine() const;
	py::buffer_info GetBufferInfo() const;
	CPyMathEngineOwner& MathEngineOwner() const { return *mathEngineOwner; }
	CDnnBlob* Blob() const { return blob.Ptr(); }
	CPyBlob Copy( const CPyMathEngine& pyMathEngine ) const;

	int DimSize(int d) const { return blob == 0 ? 0 : blob->DimSize(d); }

	py::tuple GetShape() const;
	int GetBatchLength() const { return blob == 0 ? 0 : blob->GetBatchLength(); }
	int GetBatchWidth() const { return blob == 0 ? 0 : blob->GetBatchWidth(); }
	int GetListSize() const { return blob == 0 ? 0 : blob->GetListSize(); }
	int GetHeight() const { return blob == 0 ? 0 : blob->GetHeight(); }
	int GetWidth() const { return blob == 0 ? 0 : blob->GetWidth(); }
	int GetDepth() const { return blob == 0 ? 0 : blob->GetDepth(); }
	int GetChannelsCount() const { return blob == 0 ? 0 : blob->GetChannelsCount(); }

	int GetDataSize() const { return blob == 0 ? 0 : blob->GetDataSize(); }
	int GetObjectCount() const { return blob == 0 ? 0 : blob->GetObjectCount(); }
	int GetObjectSize() const { return blob == 0 ? 0 : blob->GetObjectSize(); }

private:
	CPtr<CPyMathEngineOwner> mathEngineOwner;
	CPtr<CDnnBlob> blob;
};

void InitializeBlob( py::module& m );