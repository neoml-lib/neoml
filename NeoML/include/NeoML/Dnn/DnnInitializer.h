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

#include <NeoML/NeoMLDefs.h>
#include <NeoML/Random.h>

namespace NeoML {

class CDnnBlob;

// CDnnInitializer is the base class for initializing trainable weights of a layer
class NEOML_API CDnnInitializer : public IObject {
public:
	explicit CDnnInitializer(CRandom& _random) : random(_random) {}
	virtual void InitializeLayerParams(CDnnBlob& blob, int inputCount) = 0;

	const CRandom& Random() const { return random; }
	CRandom& Random() { return random; }

private:
	CRandom& random;
};

///////////////////////////////////////////////////////////////////////////////////////////////////////

// Initializes a blob using the Xavier algorithm (randomly chosen values from ~N(0, 1/<the input size>)
class NEOML_API CDnnXavierInitializer : public CDnnInitializer {
public:
	explicit CDnnXavierInitializer(CRandom& _random) : CDnnInitializer(_random) {}

	void InitializeLayerParams(CDnnBlob& blob, int inputCount) override;
};

///////////////////////////////////////////////////////////////////////////////////////////////////////

// Initializes a blob using uniform distribution
class NEOML_API CDnnUniformInitializer : public CDnnInitializer {
public:
	explicit CDnnUniformInitializer(CRandom& _random);
	explicit CDnnUniformInitializer(CRandom& _random, float _lowerBound, float _upperBound);

	// The lower limit of the distribution
	float GetLowerBound() const { return lowerBound; }
	void SetLowerBound(float _lowerBound) { lowerBound = _lowerBound; }

	// The upper limit of the distribution
	float GetUpperBound() const { return upperBound; }
	void SetUpperBound(float _upperBound) { upperBound = _upperBound; }

	void InitializeLayerParams(CDnnBlob& blob, int inputCount) override;

private:
	float lowerBound;
	float upperBound;
};

} // namespace NeoML
