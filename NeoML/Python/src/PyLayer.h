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
                                               
class CPyLayer {
public:
	explicit CPyLayer( CBaseLayer& _baseLayer, CPyMathEngineOwner& _mathEngineOwner ) :
		baseLayer( &_baseLayer ), mathEngineOwner( &_mathEngineOwner ) {}

	CPyMathEngineOwner& MathEngineOwner() const { return *mathEngineOwner; }

	CDnn& Dnn() const { return *baseLayer->GetDnn(); }
	IMathEngine& MathEngine() const { return mathEngineOwner->MathEngine(); }
	CBaseLayer& BaseLayer() const { return *baseLayer; }

	template<class T>
	T* Layer() const { return dynamic_cast<T*>(baseLayer.Ptr()); }

	std::string GetName() const { return std::string( baseLayer->GetName() ); }
	int GetInputCount() const { return baseLayer->GetInputCount(); }
	std::string GetInputName( int inputIdx ) const;
	int GetInputOutputIdx( int inputIdx ) const;

	void DisableLearning() { baseLayer->DisableLearning(); }
	void EnableLearning() { baseLayer->EnableLearning(); }
	bool IsLearningEnabled() const { return baseLayer->IsLearningEnabled(); }

	virtual py::object CreatePythonObject() const { NeoAssert(false); return py::cast(0); }

	void Connect( CPyLayer& layer, int outputIndex, int inputIndex );

private:
	CPtr<CPyMathEngineOwner> mathEngineOwner;
	CPtr<CBaseLayer> baseLayer;
};

std::string FindFreeLayerName( const CDnn&, const std::string& layerName, const std::string& userName );

void InitializeLayer( py::module& m );