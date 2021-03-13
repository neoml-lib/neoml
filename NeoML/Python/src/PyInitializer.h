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

#include "PyRandom.h"

class CPyInitializer {
public:
	explicit CPyInitializer( CPyRandomOwner& _randomOwner, CDnnInitializer* _initializer ) : randomOwner( &_randomOwner ), initializer( _initializer ) {}

	std::string GetClassName() const;

	CPyRandom GetRandom() const { return CPyRandom( *randomOwner.Ptr() ); }

	CPyRandomOwner& RandomOwner() const { return *randomOwner; }

	template<class T>
	T* Initializer() const { return dynamic_cast<T*>( initializer.Ptr() ); }

private:
	CPtr<CPyRandomOwner> randomOwner;
	CPtr<CDnnInitializer> initializer;
};

void InitializeInitializer( py::module& m );
