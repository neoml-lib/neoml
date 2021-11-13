/* Copyright © 2017-2021 ABBYY Production LLC

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

// Unfold layer extracts data from the regions, that would be affected by the convolution with given parameters.
//
// Input size:
//    - BD_BatchLength * BD_BatchWidth * BD_ListSize - number of images to be processed by the layer
//    - BD_Height and BD_Width are the height and the width of the images to be processed
//    - BD_Depth * BD_Channels - number of channels in the images
//
// Output size:
//    - BD_BatchLength, BD_BatchWidth, BD_ListSize remains the same
//    - BD_Height is equal to the product of the height and the width of the convolution output
//    - BD_Width and BD_Depth are equal to 1
//    - BD_Channels is equal to the number of channels in the input images, multiplied by filterHeight * filterWidth
class NEOML_API CUnfoldLayer : public CBaseLayer {
	NEOML_DNN_LAYER( CUnfoldLayer )
public:
	explicit CUnfoldLayer( IMathEngine& mathEngine );

	// Convolution parameters
	// Filter size (1 by default)
	int GetFilterHeight() const { return filterHeight; }
	void SetFilterHeight( int value );
	int GetFilterWidth() const { return filterWidth; }
	void SetFilterWidth( int value );
	// Strides (1 by default)
	int GetStrideHeight() const { return strideHeight; }
	void SetStrideHeight( int value );
	int GetStrideWidth() const { return strideWidth; }
	void SetStrideWidth( int value );
	// Paddings (0 by default)
	int GetPaddingHeight() const { return paddingHeight; }
	void SetPaddingHeight( int value );
	int GetPaddingWidth() const { return paddingWidth; }
	void SetPaddingWidth( int value );
	// Dilations (1 by default)
	int GetDilationHeight() const { return dilationHeight; }
	void SetDilationHeight( int value );
	int GetDilationWidth() const { return dilationWidth; }
	void SetDilationWidth( int value );

	void Serialize( CArchive& archive ) override;

protected:
	void Reshape() override;
	void RunOnce() override;
	void BackwardOnce() override;

private:
	int filterHeight;
	int filterWidth;
	int strideHeight;
	int strideWidth;
	int paddingHeight;
	int paddingWidth;
	int dilationHeight;
	int dilationWidth;
};

// --------------------------------------------------------------------------------------------------------------------

NEOML_API CLayerWrapper<CUnfoldLayer> Unfold( int filterHeight, int filterWidth, int strideHeight, int strideWidth,
	int paddingHeight = 0, int paddingWidth = 0, int dilationHeight = 1, int dilationWidth = 1 );

} // namespace NeoML