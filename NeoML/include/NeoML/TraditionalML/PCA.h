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
#include <NeoML/TraditionalML/Model.h>
#include <NeoMathEngine/NeoMathEngine.h>
#include <NeoML/TraditionalML/SparseFloatMatrix.h>

namespace NeoML {

// SVD solver type
enum TSvd {
	// Full svd for dense matrices
	SVD_Full = 0,
	// Randomized svd algorithm for sparse matrices
	SVD_Randomized,
	SVD_Count
};

// Computes the singular value decomposition of the `data` matrix, of shape height x width:
// `data` = `leftVectors` * `singularValues` * `rightVectors`.
// `leftVectors` is of shape height x `components`,  `rightVectors` is of shape `components` x width,
// `singularValues` contains `components` largest singular values.
// If returnLeftVectors or returnRightVectors is false then corresponding singular vectors are not returned.
// `components` is set as min(height, width) if not specified.
void NEOML_API SingularValueDecomposition( const CFloatMatrixDesc& data,
	CArray<float>& leftVectors, CArray<float>& singularValues, CArray<float>& rightVectors,
	bool returnLeftVectors = true, bool returnRightVectors = false, int components = 0 );

// Computes the singular value decomposition of the `data` matrix, of shape height x width:
// `data` = `leftVectors` * `singularValues` * `rightVectors`.
// `leftVectors` is of shape height x `components`,  `rightVectors` is of shape `components` x width,
// `singularValues` contains `components` largest singular values.
// If returnLeftVectors or returnRightVectors is false then corresponding singular vectors are not returned.
// `overSamples` - additional number of components to be calculated to ensure proper conditioning.
// `iterationCount` - number of algorithm iterations, a smaller number improves speed but can negatively impact precision.
// `seed` used to initialize a random matrix with the normal distribution for algorithm iterations.
void NEOML_API RandomizedSingularValueDecomposition( const CFloatMatrixDesc& data,
	CArray<float>& leftVectors_, CArray<float>& singularValues_, CArray<float>& rightVectors_,
	bool returnLeftVectors, bool returnRightVectors, int components,
	int iterationCount = 3, int overSamples = 10, int seed = 42 );

// PCA algorithm implementing linear dimensionality reduction
// using Singular Value Decomposition to project the data into
// a lower dimensional space
class NEOML_API CPca : public IObject {
public:
	// Determines how the number of components will be chosen
	enum TComponents {
		// Set the number of components to min(data.width, data.height)
		PCAC_None = 0,
		// Take the integer value in the Components parameter as the number of components
		PCAC_Int,
		// In case of SVD_Full number of components is selected such that
		// the value of explained_variance is greater than Components
		// 0 < Components < 1
		PCAC_Float,
        // The number of constants in the enum
		PCAC_Count
	};

	// PCA params
	struct CParams {
		TComponents ComponentsType;
		TSvd SvdSolver;
		float Components;

		CParams() :
			ComponentsType( PCAC_Int ),
			SvdSolver( SVD_Full ),
			Components( 0 )
		{
		}
	};

	CPca() = default;
	explicit CPca( const CParams& params );
	~CPca() = default;

	// Chooses the greatest singular values from `Components` and
	// selects the corresponding principal axes as the final components
	void Train( const CFloatMatrixDesc& data );
	// Transforms the data into shape ( samples x components )
	// using the principal components calculated before
	CSparseFloatMatrixDesc Transform( const CFloatMatrixDesc& data );
	// Trains and transforms the data into shape ( samples x components )
	CSparseFloatMatrixDesc TrainTransform( const CFloatMatrixDesc& data );

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
	// Matrix ( components x features ) with rows corresponding to the selected principal axis 
	CSparseFloatMatrix GetComponents();

	// Get input params
	CParams GetParams() const { return params; }
	// For serialization
	static CPtr<CPca> Create() { return FINE_DEBUG_NEW CPca(); }
	// Serializes the model
	void Serialize( CArchive& archive ) override;

private:
	CParams params;
	CArray<float> singularValues;
	CArray<float> explainedVariance;
	CArray<float> explainedVarianceRatio;
	CArray<float> componentsMatrix;
	CSparseFloatMatrix transformedMatrix;
	CSparseFloatVector meanVector;
	float noiseVariance;
	int components;

	void train( const CFloatMatrixDesc& data, bool isTransform );
	void calculateVariance( const CFloatMatrixDesc& data, const CArray<float>& s, int total_components );
	void getComponentsNum( const CArray<float>& explainedVarianceRatio, int k );
};

} // namespace NeoML
