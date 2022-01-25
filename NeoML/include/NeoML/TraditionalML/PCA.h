/* Copyright � 2017-2020 ABBYY Production LLC

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

// SVD solver type
enum TSvd {
	// Full svd for dense matrices
	SVD_Full = 0,
	// Truncated svd for sparse matrices
	SVD_Sparse,
	SVD_Count
};

// Computes the singular value decomposition of the `data` matrix, of shape m x n:
// `data` = `leftVectors` * `singularValues` * `rightVectors`.
// For SVD_Full `leftVectors` is of shape m x min(n, m), `rightVectors` is of shape min(n, m) x n,
// `singularValues` contains min(n, m) singular values.
// For SVD_Sparse parameter `components` must be set. Then `leftVectors` is of shape `components` x m,
// `rightVectors` is of shape `components` x n, `singularValues` contains `components` largest singular values.
// If returnLeftVectors or returnRightVectors is false then corresponding singular vectors are not returned.
void NEOML_API SingularValueDecomposition( const CFloatMatrixDesc& data, const TSvd& svdSolver,
	CArray<float>& leftVectors, CArray<float>& singularValues, CArray<float>& rightVectors,
	bool returnLeftVectors = true, bool returnRightVectors = false, int components = 0 );

// PCA algorithm implementing linear dimensionality reduction
// using Singular Value Decomposition to project the data into
// a lower dimensional space
class NEOML_API CPca {
public:
	// Components parameter type
	enum TComponents {
		// Set number of components as min(data.width, data.height)
		PCAC_None = 0,
		// Integer number Components representing a number of components to compute
		PCAC_Int,
		// In case of SVD_Full number of components is selected such that
		// the value of explained_variance is greater than Components
		// 0 < Components < 1
		PCAC_Float,
		PCAC_Count
	};

	// PCA params
	struct CParams {
		TComponents ComponentsType;
		TSvd SvdSolver;
		float Components;

		CParams() :
			ComponentsType( PCAC_None ),
			SvdSolver( SVD_Full ),
			Components( 0 )
		{
		}
	};

	explicit CPca( const CParams& params );
	~CPca() {};

	// Chooses `Components` greatest singular values and
	// selects the corresponding principal axis as the final components
	void Train( const CFloatMatrixDesc& data );
	// Train + transform the data into shape ( samples x components )
	CSparseFloatMatrixDesc Transform( const CFloatMatrixDesc& data );

	// Singular values corresponding to the selected principal axis
	const CArray<float>& GetSingularValues() const { return singularValues; }
	// Variance explained by each of the selected principal axis
	const CArray<float>& GetExplainedVariance() const { return explainedVariance; }
	// Percentage of variance explained by each of the selected principal axis
	const CArray<float>& GetExplainedVarianceRatio() const { return explainedVarianceRatio; }
	// Mean of singular values not corresponding to the selected principal axis
	float GetNoiseVariance() const { return noiseVariance; }
	// Selected number of principal axis
	int GetComponentsNum() const { return components; }
	// Matrix ( components x features ) with rows corresponding to the selected principal axis 
	CFloatMatrixDesc GetComponents() const {
		// not working for sparse algorithm
		NeoAssert( params.SvdSolver == SVD_Full );
		return componentsMatrix.GetDesc();
	}

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
	void calculateVariance( const CFloatMatrixDesc& data, const CArray<float>& s, int total_components );
	void getComponentsNum( const CArray<float>& explainedVarianceRatio, int k );
};

} // namespace NeoML
