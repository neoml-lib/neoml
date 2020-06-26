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
#include <NeoML/Dnn/Dnn.h>

namespace NeoML {

// CDnnImageResizeLayer implements a layer that resizes the input image
// On increasing size, the newly added pixels will be filled with defaultValue
// On decreasing size, the extra pixels will be discarded
class NEOML_API CImageResizeLayer : public CBaseLayer {
	NEOML_DNN_LAYER( CImageResizeLayer )
public:
	explicit CImageResizeLayer( IMathEngine& mathEngine );

	void Serialize( CArchive& archive ) override;

	// The side of the image where size changes
	enum TImageSide {
		IS_Left = 0, // left
		IS_Right, // right
		IS_Top, // top
		IS_Bottom, // bottom

		IS_Count,
	};

	int GetDelta( TImageSide side ) const;
	// Sets the size difference from the given side, in pixels
	void SetDelta( TImageSide side, int delta );

	float GetDefaultValue() const { return defaultValue; }
	// Sets the default value for new pixels
	void SetDefalutValue( float value ) { defaultValue = value; }

protected:
	// CBaseLayer methods
	void Reshape() override;
	void RunOnce() override;
	void BackwardOnce() override;

private:
	int deltaLeft; // the size difference from the left
	int deltaRight; // the size difference from the right
	int deltaTop; // the size difference from the top
	int deltaBottom; // the size difference from the bottom
	float defaultValue; // the default value for new pixels that appear whenever delta > 0
};

} // namespace NeoML
