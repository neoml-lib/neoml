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
#include <NeoML/TraditionalML/FloatVector.h>
#include <NeoML/TraditionalML/SparseFloatMatrix.h>
#include <NeoML/TraditionalML/ClassificationResult.h>

namespace NeoML {

// Trained classifier model interface
class NEOML_API IModel : virtual public IObject {
public:
	virtual ~IModel();

	// The number of classes
	virtual int GetClassCount() const = 0;

	// Classifies the input vector and returns true if successful, false otherwise
	virtual bool Classify( const CSparseFloatVectorDesc& data, CClassificationResult& result ) const = 0;
	virtual bool Classify( const CSparseFloatVector& data, CClassificationResult& result ) const
	{
		return Classify( data.GetDesc(), result );
	}
	virtual bool Classify( const CFloatVector& data, CClassificationResult& result ) const = 0;

	// Serializes the model
	virtual void Serialize( CArchive& archive ) = 0;
};

// Regression model for a function that returns a number
class NEOML_API IRegressionModel : virtual public IObject {
public:
	virtual ~IRegressionModel();

	// Predicts the function value on a vector
	virtual double Predict( const CSparseFloatVector& data ) const = 0;
	virtual double Predict( const CFloatVector& data ) const = 0;
	virtual double Predict( const CSparseFloatVectorDesc& desc ) const = 0;

	// Serializes the model
	virtual void Serialize( CArchive& archive ) = 0;
};

// Regression model for a function that returns a vector
class NEOML_API IMultivariateRegressionModel : virtual public IObject {
public:
	virtual ~IMultivariateRegressionModel();

	// Predicts the function value on a vector
	virtual CFloatVector MultivariatePredict( const CSparseFloatVector& data ) const = 0;
	virtual CFloatVector MultivariatePredict( const CFloatVector& data ) const = 0;

	// Serializes the model
	virtual void Serialize( CArchive& archive ) = 0;
};

//------------------------------------------------------------------------------------------------------------
// Auxiliary model registration mechanisms
// DO NOT use directly

typedef CPtr<IObject> ( *TCreateModelFunction )();

void NEOML_API RegisterModelName( const char* layerName, const std::type_info& typeInfo, TCreateModelFunction function );

void NEOML_API UnregisterModelName( const std::type_info& typeInfo );

template<class T>
class CModelClassRegistrar {
public:
	explicit CModelClassRegistrar( const char* modelName );
	~CModelClassRegistrar();

private:
	static CPtr<IObject> createModel() { return FINE_DEBUG_NEW T; }
};

template<class T>
inline CModelClassRegistrar<T>::CModelClassRegistrar( const char* modelName )
{
	RegisterModelName( modelName, typeid( T ), createModel );
}

template<class T>
inline CModelClassRegistrar<T>::~CModelClassRegistrar()
{
	UnregisterModelName( typeid( T ) );
}

//------------------------------------------------------------------------------------------------------------

// Registers a model
#define REGISTER_NEOML_MODEL( modelClassType, modelName ) static CModelClassRegistrar< modelClassType > __merge__1( _RegisterModel, __LINE__ )( modelName );

// Declares the model name
#define DECLARE_NEOML_MODEL_NAME( var, modelName ) const char* const var = modelName ;

// Retrieves the model name
NEOML_API const char* GetModelName( const IObject* model );

// Creates a registered model with a given name
NEOML_API CPtr<IObject> CreateModel( const char* modelName );

template<class T>
inline CPtr<T> CreateModel( const char* modelName )
{
	return dynamic_cast<T*>( CreateModel( modelName ).Ptr() );
}

// Polymorphic serialization of a model, registered in NeoML
template<typename TModel>
inline void SerializeModel( CArchive& archive, CPtr<TModel>& model )
{
	if( archive.IsStoring() ) {
		if( model == 0 ) {
			archive << CString();
		} else {
			CString name( NeoML::GetModelName( model ) );
			NeoAssert( name != "" );
			archive << name;
			model->Serialize( archive );
		}
	} else if( archive.IsLoading() ) {
		CString name;
		archive >> name;
		if( name == "" ) {
			model = 0;
		} else {
			model = NeoML::CreateModel<TModel>( name );
			model->Serialize( archive );
		}
	} else {
		NeoAssert( false );
	}
}

} // namespace NeoML
