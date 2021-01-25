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
#include "PyRandom.h"
#include "PyInitializer.h"
#include "PySolver.h"

class CPyDnn {
public:
	CPyDnn( CPyRandomOwner& _randomOwner, CPyMathEngineOwner& _mathEngineOwner );
	CPyDnn( CPyRandomOwner& _randomOwner, CPyMathEngineOwner& _mathEngineOwner, const std::string& path );

	py::object GetMathEngine() const;

	void SetSolver( const CPySolver& solver );
	py::object GetSolver() const;

	void SetInitializer( const CPyInitializer& initializer );
	py::object GetInitializer() const;

	py::dict GetInputs() const;
	py::dict GetOutputs() const;
	py::dict GetLayers() const;

	py::list Run( py::list inputs );

	void RunAndBackward( py::list inputs );

	void Learn( py::list inputs );

	CDnn& Dnn() const { return *dnn; }
	CPyMathEngineOwner& MathEngineOwner() const{ return *mathEngineOwner; }
	IMathEngine& MathEngine() const { return mathEngineOwner->MathEngine(); }

private:
	CPtr<CPyRandomOwner> randomOwner;
	CPtr<CPyMathEngineOwner> mathEngineOwner;
	CPyInitializer initializer;
	std::unique_ptr<CDnn> dnn;
};

void InitializeDnn( py::module& m );
