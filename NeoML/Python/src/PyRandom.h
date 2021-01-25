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

class CPyRandomOwner : public IObject {
public:
	explicit CPyRandomOwner( int seed ) : random( new CRandom( seed ) ) {}

	CRandom& Random() const { return *random; }

private:
	std::unique_ptr<CRandom> random;
};

class CPyRandom {
public:
	explicit CPyRandom( CPyRandomOwner& owner ) : randomOwner( &owner ) {}
	explicit CPyRandom( int seed ) : randomOwner( new CPyRandomOwner( seed ) ) {}
	
	CRandom& Random() const { return randomOwner->Random(); }

	CPyRandomOwner& RandomOwner() const { return *randomOwner; }

private:
	CPtr<CPyRandomOwner> randomOwner;
};

void InitializeRandom( py::module& m );
