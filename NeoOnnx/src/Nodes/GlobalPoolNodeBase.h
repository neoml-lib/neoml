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

#include "../Node.h"
#include "../NodeUtils.h"

namespace NeoOnnx {

// Global
class CGlobalPoolNodeBase : public COpNode {
public:
	CGlobalPoolNodeBase( int nodeIndex, const onnx::NodeProto& onnxNode, int opsetVersion );

protected:
	// Adds global pooling layer
	void AddPoolingLayer( TPoolingType poolingType, const CTensorDim& dimsToPool, const CTensorShape& inputShape,
		const CTensorDim& inputDim, CNeoMLLinkCache& neoMLLinks, CDnn& dnn );

private:
	void addGlobalPoolingLayer( TPoolingType poolingType, CNeoMLLinkCache& neoMLLinks, CDnn& dnn );
	void add2dPoolingLayer( TPoolingType poolingType, const CTensorShape& inputShape, const CTensorDim& inputDim, int pooledDims,
		CNeoMLLinkCache& neoMLLinks, CDnn& dnn );
	void add3dPoolingLayer( TPoolingType poolingType, const CTensorShape& inputShape, const CTensorDim& inputDim, int pooledDims,
		CNeoMLLinkCache& neoMLLinks, CDnn& dnn );
};

} // namespace NeoOnnx

