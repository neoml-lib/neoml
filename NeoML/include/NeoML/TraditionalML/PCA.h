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

class NEOML_API CPCA : public IObject {
public:
	enum TComponents {
		// Set number of components as min(data.width, data.height)
		PCAC_None = 0,
		// Integer number Components representing a number of components to compute
		PCAC_Int,
		// Number of components is selected such that
		// the value of explained_variance is greater
		// 0 < Components < 1
		PCAC_Float,
		PCAC_Count
	};

	struct CParams {
		TComponents ComponentsType;
		float Components;

		CParams() :
			ComponentsType( TComponents::PCAC_None ),
			Components( 0 )
		{
		}
	};

	explicit CPCA( const CParams& params );
	void Train( const CFloatMatrixDesc& data );
	CSparseFloatMatrixDesc Transform( const CFloatMatrixDesc& data );
	~CPCA() {};

	void GetSingularValues( CArray<float>& vals ) const { singularValues.CopyTo( vals ); }
	void GetExplainedVariance( CArray<float>& vals ) const { explainedVariance.CopyTo( vals ); }
	void GetExplainedVarianceRatio( CArray<float>& vals ) const { explainedVarianceRatio.CopyTo( vals ); }
	float GetNoiseVariance() const { return noiseVariance; }
	int GetComponentsNum() const { return components; }
	CFloatMatrixDesc GetComponents() { return componentsMatrix.GetDesc(); }

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
