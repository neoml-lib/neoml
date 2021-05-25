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

#include <NeoML/NeoML.h>

class CPyMathEngineOwner: public IObject {
public:
	CPyMathEngineOwner() : owned( false ) {}
	explicit CPyMathEngineOwner( IMathEngine* _mathEngine, bool _owned = true ) : mathEngine( _mathEngine ), owned( _owned ) {}
	~CPyMathEngineOwner() { if( !owned ) { mathEngine.release(); } }

	IMathEngine& MathEngine() const { return mathEngine.get() == 0 ? GetDefaultCpuMathEngine() : *mathEngine.get(); }

private:
	std::unique_ptr<IMathEngine> mathEngine;
	bool owned;
};

class CPyMathEngine {
public:
	explicit CPyMathEngine( CPyMathEngineOwner& owner );
	CPyMathEngine( const std::string& type, int threadCount, int index );

	std::string GetInfo() const;

	long long GetPeakMemoryUsage();
	
	void CleanUp();

	CPyMathEngineOwner& MathEngineOwner() const { return *mathEngineOwner; }

private:
	CPtr<CPyMathEngineOwner> mathEngineOwner;
};

void InitializeMathEngine(py::module& m);
