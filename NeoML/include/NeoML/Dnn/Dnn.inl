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

namespace NeoML {

inline bool CBaseLayer::IsLearningPerformed() const
{
	NeoAssert(GetDnn() != 0);
	return IsLearningNeeded() && GetDnn()->IsBackwardPerformed();
}

inline bool CBaseLayer::IsLearningNeeded() const
{
	NeoAssert(GetDnn() != 0);
	return isLearnable && isLearningEnabled && GetDnn()->IsLearningEnabled();
}

inline bool CBaseLayer::IsBackwardPerformed() const
{
	NeoAssert(GetDnn() != 0);
	return isBackwardNeeded == BS_NeedsBackward && GetDnn()->IsBackwardPerformed();
}

inline bool CBaseLayer::IsBackwardNeeded() const
{
	NeoAssert(GetDnn() != 0);
	return isBackwardNeeded == BS_NeedsBackward;
}

inline bool CBaseLayer::isInPlaceProcess() const
{
	if(inputBlobs.Size() == 0 || inputBlobs.Size() != outputBlobs.Size()) {
		return false;
	}
	return inputBlobs[0] == outputBlobs[0];
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
// CDnnLayerGraph
inline void CDnnLayerGraph::AddLayer(CBaseLayer& layer)
{
	layer.graphCount += 1;
	AddLayerImpl(layer);
}

inline void CDnnLayerGraph::DeleteLayer(const char* name)
{
	CPtr<CBaseLayer> layer = GetLayer(name);
	NeoAssert(layer.Ptr() != 0);

	DeleteLayer(*layer);
}

inline void CDnnLayerGraph::DeleteLayer(CBaseLayer& layer)
{
	CPtr<CBaseLayer> layerHold = &layer; // need to keep the reference to the layer: otherwise it may be deleted in DeleteLayerImpl

	DeleteLayerImpl(layer);

	layer.graphCount -= 1;
	NeoAssert(layer.graphCount >= 0);
}

inline void CDnnLayerGraph::DeleteAllLayers()
{
	CArray<const char*> layerList;
	GetLayerList(layerList);

	for(int i = 0; i < layerList.Size(); ++i) {
		DeleteLayer(layerList[i]);
	}
}

} // namespace NeoML
