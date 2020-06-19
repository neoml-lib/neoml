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

// Suppose each specimen we examine contains an array of objects characterized by a set of features.
// Additionally, each object is bound to the specified location on the plane.
// The neural network processes this input data and calculates some other features.
// Sometimes, you will need to use convolution; for that the data should be represented as an image on which the objects are located.
//
// This layer provides one way to create such a representation. 
// Each object is "projected" to a pixel, its features being written into this pixel channels.
// The channels for pixels that are not projections of any of the objects are filled with zeros. 
//
// This implementation assumes that the object location data is provided externally 
// as a set of already calculated indices, based on the coordinates of the pixels where the objects are projected.
//
// The layer has two inputs:
// The first tensor contains the feature values and has the dimensions (batch_size, objects_count, features_count),
// which in NeoML translates to a blob of (1, batch_size, objects_count, 1, 1, 1, features_count) dimensions
// The second tensor contains the linear indices derived from the pixel coordinates (x, y) on the target image 
// using the formula index = image_width * y + x. The blob dimensions are (1, batch_size, 1, 1, 1, 1, objects_count).
// The layer has two parameters: the target image height and width
// The output tensor has the dimensions (batch_size, image_height, image_width, features_count)
//
// This layer is intended to be used in the same situations as tensorflow.scatter_nd
class NEOML_API CPixelToImageLayer : public CBaseLayer {
	NEOML_DNN_LAYER( CPixelToImageLayer )
public:
	explicit CPixelToImageLayer( IMathEngine& mathEngine );

	void Serialize( CArchive& archive ) override;

	// The resulting image height and width
	int GetImageHeight() const { return imageHeight; }
	void SetImageHeight( int newHeight );
	int GetImageWidth() const { return imageWidth; }
	void SetImageWidth( int newWidth );

protected:
	void Reshape() override;
	void RunOnce() override;
	void BackwardOnce() override;

private:
	// the resulting image dimensions
	int imageHeight;
	int imageWidth;
	// the shift vector (to be added to the columns of the indices matrix)
	CPtr<CDnnBlob> shift;
	// the shifted indices
	CPtr<CDnnBlob> shiftedIndices;
};

// ====================================================================================================================

// The layer extracts the pixels with the specified coordinates from the specified image.
//
// It has two inputs:
// The first tensor has (batch_size, image_height, image_width, featuresCount) dimensions
// It contains the feature values along the last (Channels) dimension 
// In NeoML, the blob dimensions are (1, batch_size, 1, image_height, image_width, 1, featuresCount)
// The second tensor has (batch_size, objects_count) dimensions and contains the linear indices
// derived from the (x, y) coordinates of image pixels using the formula index = image_width * y + x
// In NeoML, the blob dimensions are (1, batch_size, 1, 1, 1, objectsCount)
// The output tensor is of the dimensions (batch_size, objectsCount, featuresCount),
// which in NeoML translates to blob dimensions (1, batch_size, objectsCount, 1, 1, 1, featuresCount)
//
// The layer is intended to be used in the same situations as tensorflow.gather_nd
class NEOML_API CImageToPixelLayer : public CBaseLayer {
	NEOML_DNN_LAYER( CImageToPixelLayer )
public:
	explicit CImageToPixelLayer( IMathEngine& mathEngine );

	void Serialize( CArchive& archive ) override;

protected:
	void Reshape() override;
	void RunOnce() override;
	void BackwardOnce() override;

private:
	// the shift vector (to be added to the columns of the indices matrix)
	CPtr<CDnnBlob> shift;
	// the shifted indices
	CPtr<CDnnBlob> shiftedIndices;
};

} // namespace NeoML
