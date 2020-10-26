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

#include <NeoML/TraditionalML/Problem.h>

namespace NeoML {

// An IProblem implementation that stores all data in memory
class NEOML_API CMemoryProblem : public IProblem {
public:
	// Creates memory problem where each vector has featureCount elements,
	// and each vector belongs to the one of the classCount classes.
	// If rowsBufferSize and elementBufferSize are greater than zero
	// and the total number of vectors is not greater than rowsBufferSize
	// and the total number of non-zero elements across all vectors is not greater than elementsBufferSize
	// then it's guaranteed that there will be no additional allocations during the CMemoryProblem::Add calls.
	CMemoryProblem( int featureCount, int classCount, int rowsBufferSize = 0, int elementsBufferSize = 0 );
	CMemoryProblem(); // used for loading serialized problems

	// Adds a vector to the set
	void Add( const CSparseFloatVectorDesc& vector, double weight, int classNumber );
	void Add( const CSparseFloatVector& vector, double weight, int classNumber );
	void Add( const CSparseFloatVectorDesc& vector, int classNumber ) { Add( vector, 1.0, classNumber ); }
	void Add( const CSparseFloatVector& vector, int classNumber ) { Add( vector, 1.0, classNumber ); }

	// Gets a vector from the set
	CSparseFloatVectorDesc GetVector( int index ) const { return matrix.GetRow( index ); }

	// Sets the feature type
	void SetFeatureType( int index, bool isDiscrete ) { isDiscreteFeature[index] = isDiscrete; }
	void SetDiscretizationValue( int index, int value );

	// Sets the vector weight
	void SetVectorWeight( int index, float weight );
	// Sets the vector class
	void SetClass( int index, int newClass );

	// IProblem interface methods:
	virtual int GetClassCount() const { return classCount; }
	virtual int GetFeatureCount() const { return featureCount; }
	virtual bool IsDiscreteFeature( int index ) const { return isDiscreteFeature[index]; }
	virtual int GetVectorCount() const { return matrix.GetHeight(); }
	virtual int GetClass( int index ) const { return classes[index]; }
	virtual CSparseFloatMatrixDesc GetMatrix() const { return matrix.GetDesc(); }
	virtual double GetVectorWeight( int index ) const { return weights[index]; };
	virtual int GetDiscretizationValue( int index ) const { return discretizationValues[index]; }

	// IObject
	virtual void Serialize( CArchive& archive );

protected:
	virtual ~CMemoryProblem() {} // delete operation prohibited

private:
	CSparseFloatMatrix matrix; // all vectors of the set
	CArray<int> classes; // the correct class labels for all the vectors
	CArray<float> weights; // the vector weights
	int classCount; // the number of classes
	int featureCount; // the number of features
	CArray<bool> isDiscreteFeature; // indicates if the feature is discrete or continuous
	CArray<int> discretizationValues; // feature sampling values
};

} // namespace NeoML
