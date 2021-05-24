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

#pragma hdrstop

#include <NeoML/NeoMLDefs.h>
#include <NeoMathEngine/NeoMathEngine.h>
#include <NeoML/TraditionalML/TrainingModel.h>

namespace NeoML {

class IPCAData : public virtual IObject {
public:
	// The number of vectors
	virtual int GetVectorCount() const = 0;

	// The number of features (vector length)
	virtual int GetFeaturesCount() const = 0;

	// Gets all input vectors as a matrix of size GetVectorCount() x GetFeaturesCount()
	virtual CFloatMatrixDesc GetMatrix() const = 0;
};

class NEOML_API CPCA : public IObject {
public:
	CPCA( NeoML::IMathEngine& mathEngine );
	virtual CPtr<IModel> Train( const IPCAData& data );
	~CPCA() {};

private:
	IMathEngine& mathEngine;
};

} // namespace NeoML
