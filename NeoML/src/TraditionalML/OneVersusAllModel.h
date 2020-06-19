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

#include <NeoML/TraditionalML/OneVersusAll.h>

namespace NeoML {

// One versus all classifier
class COneVersusAllModel : public IOneVersusAllModel {
public:
	COneVersusAllModel() {}
	explicit COneVersusAllModel( CObjectArray<IModel>& classifiers );

	// For serialization
	static CPtr<IModel> Create() { return FINE_DEBUG_NEW COneVersusAllModel(); }

	// IModel interface methods
	int GetClassCount() const override;
	bool Classify( const CSparseFloatVectorDesc& data, CClassificationResult& result ) const override;
	bool Classify( const CFloatVector& data, CClassificationResult& result ) const override;
	void Serialize( CArchive& archive ) override;

	// IOneVersusAllModel interface methods
	const CObjectArray<IModel>& GetModels() const override { return classifiers; }
	bool ClassifyEx( const CSparseFloatVector& data, COneVersusAllClassificationResult& result ) const override;
	bool ClassifyEx( const CSparseFloatVectorDesc& data, COneVersusAllClassificationResult& result ) const override;
	bool ClassifyEx( const CFloatVector& data, COneVersusAllClassificationResult& result ) const override;

protected:
	virtual ~COneVersusAllModel() {} // delete prohibited

private:
	CObjectArray<IModel> classifiers; // the binary classifiers for each of the classes in turn
};

} // namespace NeoML
