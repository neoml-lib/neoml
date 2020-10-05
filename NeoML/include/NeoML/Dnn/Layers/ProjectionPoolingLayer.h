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
// OCR Technology © 2020 ABBYY Production LLC.
// Автор:		Атрощенко Михаил
// Система: 	FineMachineLearningExt
// Описание:	Слой, осуществляющий проецирование методом усреднения по строкам/столбцам.
//				Подробнее см. статью Deep splitting and merging for table structure decomposition.

#pragma once

#include <NeoML/NeoMLDefs.h>
#include <NeoML/Dnn/Dnn.h>

namespace NeoML {

// CProjectionPoolingLayer implements a layer that calculating average along height or width
class NEOML_API CProjectionPoolingLayer : public CBaseLayer {
	NEOML_DNN_LAYER( CProjectionPoolingLayer )
public:
	// Projection direction
	enum TDirection {
		// Along BD_Width
		D_ByRows,
		// Along BD_Height
		D_ByColumns,

		D_EnumSize
	};

public:
	explicit CProjectionPoolingLayer( IMathEngine& mathEngine );
	virtual ~CProjectionPoolingLayer();

	// Projection direction
	TDirection GetDirection() const { return direction; }
	void SetDirection( TDirection _direction ) { direction = _direction; }

	bool ShouldRestoreOriginalImageSize() const { return shouldRestoreOriginalImageSize; }
	void SetRestoreOriginalImageSize( bool flag ) { shouldRestoreOriginalImageSize = flag; }

protected:
	// CBaseLayer methods
	void Serialize( CArchive& archive ) override;
	void Reshape() override;
	void RunOnce() override;
	void BackwardOnce() override;

private:
	// Projection direction
	TDirection direction;
	// Does layer preserve input blob's shape?
	// If true output blob will be of same dimension as input and pool result will be broadcasted along whole pooled dimension
	// If false pooled dimension of the output blob will be equal to 1
	// False by default
	bool shouldRestoreOriginalImageSize;

	// Temporary blob for pool results
	CPtr<CDnnBlob> projectionResultBlob;

	// Pooling descriptor
	CMeanPoolingDesc* desc;

	void initDesc();
	void destroyDesc();
};

} // namespace NeoML
