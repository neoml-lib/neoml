/* Copyright © 2017-2023 ABBYY

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
#include <NeoML/Dnn/Rowwise/RowwiseOperation.h>

namespace NeoML {

// Layer which executes chain of rowwise operations
class NEOML_API CRowwiseOperationChainLayer : public CBaseLayer {
	NEOML_DNN_LAYER( CRowwiseOperationChainLayer )
public:
	explicit CRowwiseOperationChainLayer( IMathEngine& mathEngine );
	~CRowwiseOperationChainLayer();

	// Access to the chain of operations
	int OperationCount() const { return operations.Size(); }
	const IRowwiseOperation* GetOperation( int index ) const { return operations[index]; }

	// Adds operation to the end of the chain
	void AddOperation( IRowwiseOperation* newOperation ) { operations.Add( newOperation ); }

	void Serialize( CArchive& archive ) override;

protected:
	void Reshape() override;
	void RunOnce() override;
	void BackwardOnce() override;

private:
	// Rowwise operations
	CObjectArray<IRowwiseOperation> operations;
	// MathEngine descriptors of operations in chain
	CArray<CRowwiseOperationDesc*> operationDescs;

	void deleteRowwiseDescs();
};

//=====================================================================================================================

// TODO: add this to OptimizeDnn?
void NEOML_API OptimizeRowwiseChains( CDnn& dnn, CArray<int>& chains );

} // namespace NeoML
