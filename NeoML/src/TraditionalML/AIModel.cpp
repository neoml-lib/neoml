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

#include <common.h>
#pragma hdrstop

#include <NeoML/NeoMLDefs.h>
#include <NeoML/TraditionalML/Model.h>
#include <NeoML/TraditionalML/TrainingModel.h>

namespace NeoML {

static CMap<CString, TCreateModelFunction, CDefaultHash<CString>, RuntimeHeap> registeredModels;
static CMap<const std::type_info*, CString, CDefaultHash<const std::type_info*>, RuntimeHeap> modelNames;

IModel::~IModel()
{
}

IRegressionModel::~IRegressionModel()
{
}

IMultivariateRegressionModel::~IMultivariateRegressionModel()
{
}

ITrainingModel::~ITrainingModel()
{
}

IRegressionTrainingModel::~IRegressionTrainingModel()
{
}

void RegisterModelName( const char* modelName, const std::type_info& typeInfo, TCreateModelFunction function )
{
	NeoAssert( !registeredModels.Has( modelName ) );
	registeredModels.Add( modelName, function );
	modelNames.Add( &typeInfo, modelName );
}

void UnregisterModelName( const std::type_info& typeInfo )
{
	registeredModels.Delete( modelNames.Get( &typeInfo ) );
	modelNames.Delete( &typeInfo );
}

CPtr<IObject> CreateModel( const char* modelName )
{
	TMapPosition pos = registeredModels.GetFirstPosition( modelName );
	if( pos == NotFound ) {
		return 0;
	}
	return registeredModels.GetValue( pos )();
}

const char* GetModelName( const IObject* model )
{
	if( model == 0 ) {
		return "";
	}
	const std::type_info& modelType = typeid( *model );
	TMapPosition pos = modelNames.GetFirstPosition( &modelType );
	if( pos == NotFound ) {
		return "";
	}
	return modelNames.GetValue( pos );
}

} // namespace NeoML
