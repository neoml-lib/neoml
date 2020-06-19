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

// The lookup layer that accepts a set of indexes and returns the sum of their vector representations
// The input has BatchLength x BatchWidth x ObjectSize size
// The output has BatchLength x BatchWidth x 1 x 1 x 1 x VectorSize size
// If you pass a negative index no vector will be added in that place 
// (implemented to make it possible taking a sum of objects with different numbers of vectors)
class NEOML_API CAccumulativeLookupLayer : public CBaseLayer {
	NEOML_DNN_LAYER( CAccumulativeLookupLayer )
public:
	explicit CAccumulativeLookupLayer( IMathEngine& mathEngine );

	void Serialize( CArchive& archive ) override;

	// The size and number of vector representations
	void SetDimension( const CLookupDimension& newDimension );
	const CLookupDimension& GetDimension() const;

	// Access the representations
	CPtr<CDnnBlob> GetEmbeddings() const;
	void SetEmbeddings( const CPtr<CDnnBlob>& newEmbeddings );

protected:
	// CBaseLayer methods
	void Reshape() override;
	void RunOnce() override;
	void BackwardOnce() override;
	void LearnOnce() override;

private:
	CLookupDimension lookupDimension; // The size of representations table
};

} // namespace NeoML
