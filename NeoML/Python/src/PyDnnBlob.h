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

class CPyDnnBlob {
public:
	CPyDnnBlob( const CPyMathEngine& pyMathEngine, TBlobType type, int batchLength, int batchWidth, int listSize,
		int height, int width, int depth, int channels );
	CPyDnnBlob( const CPyMathEngine& pyMathEngine, TBlobType blobType, int batchLength, int batchWidth, int listSize,
		int height, int width, int depth, int channels, py::array data );
	CPyDnnBlob( const CPyMathEngine& pyMathEngine, py::array data );
	CPyDnnBlob( CPyMathEngineOwner& _mathEngineOwner, CDnnBlob& _blob );
	CPyDnnBlob( const CPyMathEngine& pyMathEngine, const std::string& path );

	int DimSize(int d) const { return blob->DimSize(d); }

	py::tuple GetShape() const;
	int GetBatchLength() const { return blob->GetBatchLength(); }
	int GetBatchWidth() const { return blob->GetBatchWidth(); }
	int GetListSize() const { return blob->GetListSize(); }
	int GetObjectCount() const { return blob->GetObjectCount(); }
	int GetHeight() const { return blob->GetHeight(); }
	int GetWidth() const { return blob->GetWidth(); }
	int GetDepth() const { return blob->GetDepth(); }
	int GetChannelsCount() const { return blob->GetChannelsCount(); }
	int GetDataSize() const { return blob->GetDataSize(); }
	int GetObjectSize() const { return blob->GetObjectSize(); }
	int GetGeometricalSize() const { return blob->GetGeometricalSize(); }

	py::array GetData() const { return CreateArray( *blob ); }

	CPyMathEngineOwner& MathEngineOwner() const { return *mathEngineOwner; }
	CDnnBlob& Blob() const { return *blob; }

private:
	CPtr<CPyMathEngineOwner> mathEngineOwner;
	CPtr<CDnnBlob> blob;
};

void InitializeBlob( py::module& m );