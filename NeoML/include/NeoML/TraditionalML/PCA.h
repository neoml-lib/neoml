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
#include <NeoMathEngine/NeoMathEngine.h>
#include <NeoML/TraditionalML/SparseFloatMatrix.h>

namespace NeoML {

// Principal component analysis (PCA) algorithm.
// It uses singular value decomposition to project the data into
// a lower dimensional space.
class NEOML_API CPca {
public:
	// Determines how the number of components will be chosen
	enum TComponents {
		// Set the number of components to min(data.width, data.height)
		PCAC_None = 0,
		// Take the integer value in the Components field for the number of components
		PCAC_Int,
		// Select the number of components so that the value of explained_variance 
        // is greater than the float value in the Components field
		// In this case 0 < Components < 1
		PCAC_Float,
        // The number of constants in the enum
		PCAC_Count
	};

	// PCA params
	struct CParams {
		TComponents ComponentsType;
		float Components;

		CParams() :
			ComponentsType( PCAC_None ),
			Components( 0 )
		{
		}
	};

	explicit CPca( const CParams& params );
	~CPca() {};

	// Chooses the greatest singular values from `Components` and
	// selects the corresponding principal axes as the final components
	void Train( const CFloatMatrixDesc& data );
	// Trains and transforms the data into shape ( samples x components )
	CSparseFloatMatrixDesc Transform( const CFloatMatrixDesc& data );

	// Singular values corresponding to the selected principal axes
	const CArray<float>& GetSingularValues() const { return singularValues; }
	// Variance explained by each of the selected principal axes
	const CArray<float>& GetExplainedVariance() const { return explainedVariance; }
	// Percentage of variance explained by each of the selected principal axis
	const CArray<float>& GetExplainedVarianceRatio() const { return explainedVarianceRatio; }
	// Mean of singular values not corresponding to the selected principal axes
	float GetNoiseVariance() const { return noiseVariance; }
	// Selected number of principal axes
	int GetComponentsNum() const { return components; }
	// Matrix ( components x features ) with rows corresponding to the selected principal axes 
	CFloatMatrixDesc GetComponents() const { return componentsMatrix.GetDesc(); }

private:
	const CParams params;
	CArray<float> singularValues;
	CArray<float> explainedVariance;
	CArray<float> explainedVarianceRatio;
	CSparseFloatMatrix componentsMatrix;
	CSparseFloatMatrix transformedMatrix;
	float noiseVariance;
	int components;

	void train( const CFloatMatrixDesc& data, bool isTransform );
	void calculateVariance(  IMathEngine& mathEngine, const CFloatHandle& s, int m, int k );
	void getComponentsNum( const CArray<float>& explainedVarianceRatio, int k );
};

} // namespace NeoML
