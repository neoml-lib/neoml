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

#include <NeoML/TraditionalML/OneVersusOne.h>

namespace NeoML {

// One versus one classifier
class COneVersusOneModel : public IModel {
public:
	COneVersusOneModel() {}
	explicit COneVersusOneModel( CObjectArray<IModel>& classifiers );

	// For serialization
	static CPtr<IModel> Create() { return FINE_DEBUG_NEW COneVersusOneModel(); }

	// IModel interface methods
	int GetClassCount() const override { return classCount; }
	bool Classify( const CFloatVectorDesc& data, CClassificationResult& result ) const override;
	bool Classify( const CFloatVector& data, CClassificationResult& result ) const override
		{ return Classify( data.GetDesc(), result ); }
	void Serialize( CArchive& archive ) override;

protected:
	virtual ~COneVersusOneModel() = default; // disable public delete

private:
	CObjectArray<IModel> classifiers; // binary classifiers for each pair of classes
	int classCount; // number of classes
};

} // namespace NeoML
