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

#include <NeoML/NeoMLDefs.h>
#include <NeoML/Dnn/Dnn.h>

namespace NeoML {

// Converts an enumeration to a binarized vector form
class NEOML_API CEnumBinarizationLayer : public CBaseLayer {
	NEOML_DNN_LAYER( CEnumBinarizationLayer )
public:
	explicit CEnumBinarizationLayer( IMathEngine& mathEngine );

	void Serialize( CArchive& archive ) override;

	// Enumeration size
	void SetEnumSize(int _enumSize);
	int GetEnumSize() const { return enumSize; }

protected:
	// CBaseLayer methods
	void Reshape() override;
	void RunOnce() override;
	void BackwardOnce() override;

private:
	// The size
	int enumSize;
};

//-------------------------------------------------------------------------------------------------

// Converts a bitset represented by bits in the Channels dimension to a vector containing values 0 and 1
// The Channels dimension of the result is equal to BitSetSize
class NEOML_API CBitSetVectorizationLayer : public CBaseLayer {
	NEOML_DNN_LAYER( CBitSetVectorizationLayer )
public:
	explicit CBitSetVectorizationLayer( IMathEngine& mathEngine );

	void Serialize( CArchive& archive ) override;

	// Bitset size
	void SetBitSetSize( int _bitSetSize );
	int GetBitSetSize() const { return bitSetSize; }

protected:
	// CBaseLayer methods
	void Reshape() override;
	void RunOnce() override;
	void BackwardOnce() override;

private:
	// The size
	int bitSetSize;
};

} // namespace NeoML
