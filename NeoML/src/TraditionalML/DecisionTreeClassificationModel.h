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

#include <DecisionTreeNodeBase.h>

namespace NeoML {

// A node in the decision tree for classification
class CDecisionTreeClassificationModel : public CDecisionTreeNodeBase, public IDecisionTreeModel {
public:
	// For serialization
	static CPtr<IModel> Create() { return FINE_DEBUG_NEW CDecisionTreeClassificationModel(); }

	// IDecisionTreeModel interface methods
	int GetChildrenCount() const override;
	CPtr<IDecisionTreeModel> GetChild( int index ) const override;
	void GetNodeInfo( CDecisionTreeNodeInfo& result ) const override;

	// IModel interface methods
	int GetClassCount() const override;
	bool Classify( const CSparseFloatVectorDesc& data, CClassificationResult& result ) const override;
	bool Classify( const CFloatVector& data, CClassificationResult& result ) const override;
	void Serialize( CArchive& archive ) override;

private:
	bool classify( CDecisionTreeNodeBase* node, CClassificationResult& result ) const;
};

} // namespace NeoML
