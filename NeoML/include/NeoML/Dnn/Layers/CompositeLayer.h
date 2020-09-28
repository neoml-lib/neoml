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
#include <NeoML/Dnn/Layers/BackLinkLayer.h>

namespace NeoML {

class CCompositeSourceLayer;
class CCompositeSinkLayer;

// CCompositeLayer implements a layer that contains a sub-network of other layers
class NEOML_API CCompositeLayer : public CBaseLayer, public CDnnLayerGraph {
	NEOML_DNN_LAYER( CCompositeLayer )
public:
	explicit CCompositeLayer( IMathEngine& mathEngine, const char* name = nullptr );

	void Serialize( CArchive& archive ) override;

	// The mapping of the internal layer inputs to the composite layer inputs
	void SetInputMapping(int inputNumber, const char* internalLayerName, int internalLayerInput = 0);
	void SetInputMapping(int inputNumber, CBaseLayer& internalLayer, int internalLayerInput = 0);
	void SetInputMapping(const char* internalLayerName) { SetInputMapping(0, internalLayerName, 0); }
	void SetInputMapping(CBaseLayer& internalLayer) { SetInputMapping(0, internalLayer, 0); }

	// The mapping of the internal layer outputs to the composite layer outputs
	void SetOutputMapping(int outputNumber, const char* internalLayerName, int internalLayerOutput = 0);
	void SetOutputMapping(int outputNumber, CBaseLayer& internalLayer, int internalLayerOutput = 0)
		{ SetOutputMapping(outputNumber, internalLayer.GetName(), internalLayerOutput); }
	void SetOutputMapping(const char* internalLayerName) { SetOutputMapping(0, internalLayerName, 0); }
	void SetOutputMapping(CBaseLayer& internalLayer) { SetOutputMapping(internalLayer.GetName()); }

	// Internal logging settings
	void EnableInternalLogging() { areInternalLogsEnabled = true; }
	void DisableInternalLogging() { areInternalLogsEnabled = false; }
	bool AreInternalLogsEnabled() const { return areInternalLogsEnabled; }

	// Access to the internal layers
	int GetLayerCount() const override { return layers.Size(); }
	void GetLayerList(CArray<const char*>& layerList) const override;
	CPtr<CBaseLayer> GetLayer(const char* name) override;
	CPtr<const CBaseLayer> GetLayer(const char* name) const override;
	bool HasLayer(const char* name) const override;

	// Returns the total size of the output blobs
	size_t GetOutputBlobsSize() const override;

	// Returns the total size of trainable parameters
	size_t GetTrainableParametersSize() const override;

	// Starts processing a new sequence
	void RestartSequence() override;

protected:
	virtual ~CCompositeLayer();

	void Reshape() override;
	void RunOnce() override;
	void BackwardOnce() override;
	void LearnOnce() override;
	void OnDnnChanged( CDnn* ) override;
	void FilterLayerParams( float threshold ) override;
	
	// The network object for the internal layers
	const CDnn* GetInternalDnn() const { return internalDnn; }
	CDnn* GetInternalDnn() { return internalDnn; }
	
	// The source layers for the internal network (created only when it starts)
	const CCompositeSourceLayer* Source(int i) const { return sources[i]; }
	CCompositeSourceLayer* Source(int i) { return sources[i]; }
	int GetSourceCount() const { return sources.Size(); }
	// The sink layers for the internal network
	const CCompositeSinkLayer* Sink(int i) const { return sinks[i]; }
	CCompositeSinkLayer* Sink(int i) { return sinks[i]; }
	int GetSinkCount() const { return sinks.Size(); }
	// Deletes the network sources or sinks
	void DeleteAllSources();
	void DeleteAllSinks();

	// The output mapping description
	struct COutputMapping {
		CString InternalLayerName; 
		int InternalLayerOutput;
		COutputMapping() : InternalLayerOutput(0) {}
	};
	const COutputMapping& GetOutputMapping( int i ) const { return outputMappings[i]; }

	// Internal network run and backpropagation (the methods should be overloaded in the derived classes)
	virtual void RunInternalDnn();
	virtual void RunInternalDnnBackward();
	// Sets the internal network parameters
	virtual void SetInternalDnnParams();

private:
	// Adds/deletes a layer
	void AddLayerImpl(CBaseLayer& layer) override;
	void DeleteLayerImpl(CBaseLayer& layer) override;

	// The internal network object
	CDnn* internalDnn;

	// The layer map
	CObjectArray<CBaseLayer> layers;
	CMap<CString, CBaseLayer*> layerMap;

	// The internal network sources (only created when the network starts)
	CObjectArray<CCompositeSourceLayer> sources;
	// The internal network sinks
	CObjectArray<CCompositeSinkLayer> sinks;
	// The output mappings
	CArray<COutputMapping> outputMappings;
	
	// Indicates if the internal network logging is enabled
	bool areInternalLogsEnabled;

	void processBackwardOrLearn();

	// Gets the name of the source/sink with the given number
	// Used to then connect the internal layer to it
	CString getSourceName(int num) const;
	CString getSinkName(int num) const;

	void createSources();
	void createSinks();
	void setInputDescs();
	void setOutputDescs();
	void setInputBlobs();
	void setOutputBlobs();

	// Indicates that the layer is composite (contains a sub-network)
	bool isComposite() const override { return true; }
	// The hook for inserting the child data in the archive, for consistency
	virtual void serializationHook( CArchive& archive );
};

///////////////////////////////////////////////////////////////////////////////////////////////////////

// CCompositeSourceLayer implements a source for the composite layer
// Accepts a blob of data to pass to the internal network
class NEOML_API CCompositeSourceLayer : public CBaseLayer {
	NEOML_DNN_LAYER( CCompositeSourceLayer )
public:
	explicit CCompositeSourceLayer( IMathEngine& mathEngine ) :
		CBaseLayer( mathEngine, "CCnnCompositeSourceLayer", false ), desc(CT_Float) {}

	// Sets the input data blob description
	void SetBlobDesc( const CBlobDesc& _desc );
	// Sets the input blob
	void SetBlob(CDnnBlob* blob);
	// Retrieves a reference to the blob
	const CPtr<CDnnBlob>& GetBlob() const { return blob; }

	const CPtr<CDnnBlob>& GetDiffBlob() const { return diffBlob; }
	// Sets the blob with corrections
	virtual void SetDiffBlob(CDnnBlob* blob);

	void Serialize( CArchive& archive ) override;

protected:
	// CBaseLayer methods
	void Reshape() override;
	void RunOnce() override;
	void BackwardOnce() override;
	void AllocateOutputBlobs() override;

private:
	CBlobDesc desc;
	CPtr<CDnnBlob> blob;
	CPtr<CDnnBlob> diffBlob;
};

///////////////////////////////////////////////////////////////////////////////////////////////////////

// CCompositeSinkLayer implements a sink for the composite layer
// It is used to pass the internal network outputs to the composite layer outputs
class NEOML_API CCompositeSinkLayer : public CBaseLayer {
	NEOML_DNN_LAYER( CCompositeSinkLayer )
public:
	explicit CCompositeSinkLayer( IMathEngine& mathEngine ) : CBaseLayer( mathEngine, "CCnnCompositeSinkLayer", false ) {}

	// Retrieves the reference to the input blob; may only be called after Reshape of the internal network
	const CBlobDesc& GetInputDesc() const { return inputDescs[0]; }
	// Retrieves the reference to the input blob; may only be called from RunOnce of the internal network
	const CPtr<CDnnBlob>& GetInputBlob() const { return blob; }
	void FreeInputBlob() { Reshape(); }

	// Retrieves or sets the blob with corrections
	const CPtr<CDnnBlob>& GetDiffBlob() const { return diffBlob; }
	virtual void SetDiffBlob( CDnnBlob* blob );

	void Serialize( CArchive& archive ) override;

protected:
	CPtr<CDnnBlob> blob;
	CPtr<CDnnBlob> parentBlob;
	CPtr<CDnnBlob> diffBlob;

	void Reshape() override;
	void RunOnce() override;
	void BackwardOnce() override;
};

} // namespace NeoML
