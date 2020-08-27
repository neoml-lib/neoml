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
#include <NeoML/Random.h>
#include <NeoML/Dnn/DnnSolver.h>
#include <NeoML/Dnn/DnnInitializer.h>
#include <NeoMathEngine/MemoryHandle.h>
#include <NeoMathEngine/NeoMathEngine.h>
#include <NeoML/Dnn/DnnBlob.h>
#include <stdint.h>

namespace NeoML {

// The macros for the internal name of a NeoML layer
// If this macros is used when declaring a class, that class may be registered as a NeoML layer
#define NEOML_DNN_LAYER( className ) friend class CLayerClassRegistrar< className >;

// Registers the class as a NeoML layer
#define REGISTER_NEOML_LAYER( classType, name ) static CLayerClassRegistrar< classType > __merge__1( _RegisterLayer, __LINE__ )( name, 0 );
#define REGISTER_NEOML_LAYER_EX( classType, name1, name2 ) static CLayerClassRegistrar< classType > __merge__1( _RegisterLayer, __LINE__ )( name1, name2 );

typedef CPtr<CBaseLayer> ( *TCreateLayerFunction )( IMathEngine& mathEngine );

void NEOML_API RegisterLayerName( const char* mainName, const char* additionalName, const std::type_info& typeInfo, TCreateLayerFunction function );

void NEOML_API UnregisterLayerName( const std::type_info& typeInfo );

//------------------------------------------------------------------------------------------------------------

template<class T>
class CLayerClassRegistrar {
public:
	CLayerClassRegistrar( const char* mainName, const char* additionalName );
	~CLayerClassRegistrar();

private:
	static CPtr<CBaseLayer> createObject( IMathEngine& mathEngine ) { return FINE_DEBUG_NEW T( mathEngine ); }
};

template<class T>
inline CLayerClassRegistrar<T>::CLayerClassRegistrar( const char* mainName, const char* additionalName )
{
	RegisterLayerName( mainName, additionalName, typeid( T ), createObject );
}

template<class T>
inline CLayerClassRegistrar<T>::~CLayerClassRegistrar()
{
	UnregisterLayerName( typeid( T ) );
}

class CDnn;
class CDnnLayerGraph;

///////////////////////////////////////////////////////////////////////////////////////////////////////

// CBaseLayer is the base class for all layers with which the network can function. 
// Each layer has a string name that should be unique in the network. Each layer may have 
// one or several inputs and one or several outputs.
class NEOML_API CBaseLayer : public virtual IObject {
public:
	CBaseLayer( IMathEngine& mathEngine, const char* name, bool isLearnable );

	// The current network (described by a CDnn class) to which the layer belongs
	// While a layer is connected to a network, you may not change its basic configuration,
	// such as its name, the list of inputs, the size of a convolution window, etc.
	// While a layer belongs to a network, only the settings like input blob size 
	// or function coefficients (for example, during training) may be changed
	const CDnn* GetDnn() const { return dnn; }
	CDnn* GetDnn() { return dnn; }

	// Gets the layer name
	const char* GetName() const { return name; }
	virtual void SetName( const char* _name );

	// Connects this layer's inputNumber input to the specified layer's outputNumber output
	virtual void Connect( int inputNumber, const char* layer, int outputNumber = 0 );
	void Connect( int inputNumber, const CBaseLayer& layer, int outputNumber = 0 ) { Connect(inputNumber, layer.GetName(), outputNumber); }
	void Connect(const char* input) { Connect(0, input); }
	void Connect(const CBaseLayer& layer) { Connect(0, layer.GetName()); }

	// Gets the number of layer inputs
	int GetInputCount() const { return inputs.Size(); }
	// Gets the input description
	const char* GetInputName(int number) const { return inputs[number].Name; }
	int GetInputOutputNumber(int number) const { return inputs[number].OutputNumber; }

	// The output descriptions may be obtained after the network has been built, for example, in the Reshape method call
	// Gets the number of layer outputs
	int GetOutputCount() const { return outputs.Size(); }

	// Indicates that the layer parameters may be trained
	bool IsLearnable() const { return isLearnable; }
	// Learning
	void DisableLearning();
	void EnableLearning();
	bool IsLearningEnabled() const { return isLearningEnabled; }

	// Base learning rate (the learning strategy may change 
	// the relative learning rates inside the network, but the base rate stays the same)
	float GetBaseLearningRate() const { return baseLearningRate; }
	void SetBaseLearningRate( float rate ) { baseLearningRate = rate; }
	// The base regularization multiplier
	float GetBaseL2RegularizationMult() const { return baseL2RegularizationMult; }
	void SetBaseL2RegularizationMult( float mult ) { baseL2RegularizationMult = mult; }
	float GetBaseL1RegularizationMult() const { return baseL1RegularizationMult; }
	void SetBaseL1RegularizationMult(float mult) { baseL1RegularizationMult = mult; }
	// Begins processing a new sequence
	// The method is overloaded for the composite layer and the backward link layer
	virtual void RestartSequence() {} 

	virtual void Serialize(CArchive& archive);

	// Indicates that backpropagation should be performed for the layer 
	// even if there are no trainable layers before it
	bool GetBackwardForced() const { return isBackwardForced; }
	void SetBackwardForced(bool forced);

	// Returns the total size of all output blobs together
	virtual size_t GetOutputBlobsSize() const;

	// Returns the total size of trainable parameters in this layer
	// Returns the total size of trainable parameters of its internal layers, if layer is composite or recurrent
	virtual size_t GetTrainableParametersSize() const;

protected:
	// A virtual method that creates output blobs using the input blobs
	virtual void Reshape() = 0;
	// A virtual method that implements one step of a forward pass
	virtual void RunOnce() = 0;
	// A virtual method that implements one step of a backward pass
	virtual void BackwardOnce() = 0;
	// A virtual method that implements one learning step
	virtual void LearnOnce();
	// Indicates that learning must be performed for the layer on the current step
	bool IsLearningPerformed() const;
	// Indicates that learning must be performed for the layer when Learn method is called
	bool IsLearningNeeded() const;
	// Indicates that backpropagation should be performed for the layer on the current step
	bool IsBackwardPerformed() const;
	// Indicates that backpropagation must be performed for the layer when Learn method is called
	bool IsBackwardNeeded() const;
	// Gets a pointer to the layer connected to the given input
	CBaseLayer* GetInputLayer(int input) { return inputLinks[input].Layer; }
	const CBaseLayer* GetInputLayer(int input) const { return inputLinks[input].Layer; }
	// Checks if the layer has inputs
	void CheckInputs() const;
	// Checks if the layer has only one input
	void CheckInput1() const;
	// Checks if the layer has outputs
	void CheckOutputs() const;
	// Registers the blob with the data needed for learning or backpropagation
	void RegisterRuntimeBlob(CPtr<CDnnBlob>& blob);

	// Layer input descriptions
	CArray<CBlobDesc> inputDescs;
	// Layer output descriptions
	CArray<CBlobDesc> outputDescs;

	// Input and output blobs
	CObjectArray<CDnnBlob> inputBlobs;
	CObjectArray<CDnnBlob> outputBlobs;

	// Input diff - the blobs with simulated errors for input layers training
	CObjectArray<CDnnBlob> inputDiffBlobs;
	// Output diff - the blobs with simulated errors for learning
	CObjectArray<CDnnBlob> outputDiffBlobs;

	// The blobs for trainable parameters
	CObjectArray<CDnnBlob> paramBlobs;
	// The blobs where the parameter diffs are stored
	CObjectArray<CDnnBlob> paramDiffBlobs;

	// Initializes the parameters blob using the specified initializing algorithm
	// If inputSize == 0, the blob will have the (inputBlobs[input] / 2) size
	// (halving the matrix size means that we assume about half the neurons will be 0)
	void InitializeParamBlob( int input, CDnnBlob& blob, int inputSize = 0 );
	// Called by a layer to warn that Reshape should be done before the next run
	void ForceReshape();

	virtual void OnDnnChanged( CDnn* ) {}

	void SetOutputBlob(int num, CDnnBlob* blob);

	// Indicates if the layer may be used for in-place processing (the output blobs replace the input blobs)
	// May be called from Reshape method
	bool IsInPlaceProcessAvailable() const;

	// Fills with zeros the parameters that are less (but not equal) than a given threshold
	virtual void FilterLayerParams( float /*threshold*/ ) {}

	// Retrieves the reference to the IMathEngine with which the layer was created
	IMathEngine& MathEngine() const;

	// Allocates the output blobs
	// The default implementation creates the outputBlobs array using the output descriptions
	virtual void AllocateOutputBlobs();

private:
	// The link between two layers, connecting one layer output to another layer input
	struct CDnnLayerLink {
		// Output and input links
		CBaseLayer* Layer; // the pointer to the linked layer
		int OutputNumber;	// the number of the output to which the connection leads
	};

	// Describes an input connection
	struct CInputInfo {
		CString Name; // the name of the layer that is connected to the input
		int OutputNumber; // the number of that layer's output that is connected to the input
	
		CInputInfo() { OutputNumber = NotFound; }
	};

	IMathEngine& mathEngine; 	// the layer's MathEngine
	CString name;				// the layer name
	CDnn* dnn;					// the pointer to the current network; may be null if the layer does not belong to a network
	CArray<CInputInfo> inputs;	// inputs list

	// Indicates if the layer may be trained
	const bool isLearnable;

	// Indicates if learning is enabled for the layer
	bool isLearningEnabled;
	// The base learning rate (may vary inside the network depending on the learning strategy)
	float baseLearningRate;
	// Base regularization multiplier (may vary inside the network depending on the learning strategy)
	float baseL2RegularizationMult;
	float baseL1RegularizationMult;
	// Indicates if backpropagation should be performed for the layer
	enum TBackwardStatus {
		BS_Unknown,
		BS_NeedsBackward,
		BS_DoesntNeedBackward
	};
	TBackwardStatus isBackwardNeeded;
	// Forces backpropagation
	bool isBackwardForced;
	// Forces reshaping the layer even with unchanged inputs
	// May be useful if you change the parameters that determine the output size
	bool forcedReshape;

	// Input layer links
	CArray<CDnnLayerLink> inputLinks;
	// The number of connections to each layer output
	CArray<int> outputs;
	// The number of times each output was processed
	CArray<int> outputProcessedCount;

	// Indicates if the layer should be reshaped
	bool isReshapeNeeded;
	// The number of the last network run. As a layer may be called several times 
	// during RunOnce method execution, it will first check the run number 
	// and do no calculations if it is still the same run
	int lastRunNumber;

	// The number of output diffs ready for backpropagation
	// When the ready diffs and the outputs numbers become the same, the layer is ready for backpropagation
	// (BackwardRunAndLearnOnce method may be called for this layer)
	CArray<int> readyOutputDiffs;

	// The temporary data for backpropagation or learning
	CObjectArray<CDnnBlob> runtimeBlobs;
	CArray<CPtr<CDnnBlob>*> runtimeBlobPtrs;

	// The temporary blob cache for sequence processing in a recurrent layer
	enum TBlobCacheType {
		BCT_Input,
		BCT_Output,
		BCT_Runtime,

		BCT_Count
	};
	CObjectArray<CDnnBlob> blobCache[BCT_Count];

	// The number of graphs with which the layer is connected
	int graphCount;

	// Switches the specified blobs into sequence processing mode
	void switchBlobsToSequentialMode(CObjectArray<CDnnBlob>& blobs, TBlobCacheType cacheType, bool storeParent);
	CDnnBlob* switchBlobToSequentialMode(CDnnBlob* blob, TBlobCacheType cacheType, bool storeParent);
	void switchBlobsToNonSequentialMode(CObjectArray<CDnnBlob>& blobs, TBlobCacheType cacheType, bool clear);
	CDnnBlob* switchBlobToNonSequentialMode(CDnnBlob* blob);
	void clearAllRuntimeBlobs();

	// Clones a blob to store diffs
	CDnnBlob* cloneBlobForDiff(CDnnBlob* blob);

	// Indicates if the layer uses in-place processing (the output blobs replace the input blobs)
	bool isInPlaceProcess() const;
	// Indicates if the layer is composite (contains another sub-network)
	virtual bool isComposite() const { return false; }

	//////////////////////////////////////////////////////////////////////////////////////////////////
	// The methods and data for interacting with the network

	void setDnn( CDnn* newDnn );
	void link();
	void addOutput(int number);
	void unlink();
	void reshape();
	void setInputDesc(int i);
	void runOnce();
	void recheckBackwardNeeded();
	void backwardRunAndLearnOnce();
	void transferDiffBlob( CDnnBlob* diffBlob, int outputNum );
	void onOutputProcessed( int index );

	friend class CDnn;
	friend class CDnnLayerGraph;
	friend class CDnnSolver;
};

///////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////
// CDnnLayerGraph is the base class for a layer graph
class NEOML_API CDnnLayerGraph {
public:
	virtual ~CDnnLayerGraph() {}

	// Accessing the layers
	virtual int GetLayerCount() const = 0;
	virtual void GetLayerList( CArray<char const*>& layerList ) const = 0;
	virtual CPtr<CBaseLayer> GetLayer( const char* name ) = 0;
	virtual CPtr<const CBaseLayer> GetLayer( const char* name ) const = 0;
	virtual bool HasLayer( const char* name ) const = 0;

	void AddLayer(CBaseLayer& layer);
	void DeleteLayer(const char* name);
	void DeleteLayer(CBaseLayer& layer);
	void DeleteAllLayers();

protected:
	// Adds and deletes a layer
	virtual void AddLayerImpl( CBaseLayer& layer ) = 0;
	virtual void DeleteLayerImpl( CBaseLayer& layer ) = 0;
};

///////////////////////////////////////////////////////////////////////////////////////////////////////

// Gets the NeoML exception handler for math engine
NEOML_API IMathEngineExceptionHandler* GetExceptionHandler();

// Turns the single-thread mode on and off
// It affects the default math engine to be used: 
// GetSingleThreadCpuMathEngine() in single-thread mode, GetMultiThreadCpuMathEngine() otherwise
void NEOML_API EnableSingleThreadMode( bool enable );
bool NEOML_API IsSingleThreadModeOn();

// Gets a math engine that performs calculations on CPU
// GetSingleThreadCpuMathEngine() or GetMultiThreadCpuMathEngine() is used
// The engine does not need to be destroyed
NEOML_API IMathEngine& GetDefaultCpuMathEngine();

// Gets a math engine that uses one CPU thread
// The engine does not need to be destroyed
NEOML_API IMathEngine& GetSingleThreadCpuMathEngine();

// Gets a math engine that performs calculations on CPU using 
// the recommended for this CPU number of threads
// The engine does not need to be destroyed
NEOML_API IMathEngine& GetMultiThreadCpuMathEngine();

// Gets a math engine that works on the recommended GPU
// Set memoryLimit = 0 to use all available memory
// Returns 0 if the GPU is not available
// The engine SHOULD be destroyed after use with standart delete
NEOML_API IMathEngine* GetRecommendedGpuMathEngine( size_t memoryLimit );

///////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////
// CDnn class represents a neural network
class NEOML_API CDnn : public CDnnLayerGraph {
public:
	CDnn( CRandom& random, IMathEngine& mathEngine );
	~CDnn();

	// Sets a text stream for logging processing
	// By default logging is off (set to null to turn off)
	CTextStream* GetLog() { return log; }
	void SetLog( CTextStream* newLog ) { log = newLog; }

	// Sets the logging frequence (by default, each 100th Run or RunAndLearn call is recorded)
	int GetLogFrequency() const { return logFrequency; }
	void SetLogFrequency(int _logFrequency) { logFrequency = _logFrequency; }
	// Indicates if the current run is logged
	bool IsLogging() const { return log != 0 && runNumber % logFrequency == 0; }

	// Accessing the layers
	int GetLayerCount() const override { return layerMap.Size(); }
	void GetLayerList( CArray<const char*>& layerList ) const override;
	CPtr<CBaseLayer> GetLayer( const char* name ) override;
	CPtr<const CBaseLayer> GetLayer( const char* name ) const override;
	bool HasLayer( const char* name ) const override { return layerMap.Has( name ); }

	// Runs the network: all data from the input blobs is used
	void RunOnce();
	// Runs the network and performs a backward pass with the input data
	void RunAndBackwardOnce();
	// Runs the network, performs a backward pass and updates the trainable weights
	void RunAndLearnOnce();

	// Gets the maximum sequence length
	int GetMaxSequenceLength() const { return maxSequenceLength; }
	// Gets the current position in sequence (makes sense when calling from one of the Run... methods)
	int GetCurrentSequencePos() const { return currentSequencePos; }
	// Indicates if the sequence is processed in reverse order
	bool IsReverseSequense() const { return isReverseSequense; }
	// Indicates if the current position is the first in sequential processing
	bool IsFirstSequencePos() const { return GetCurrentSequencePos() == (IsReverseSequense() ? GetMaxSequenceLength() - 1 : 0); }
	// Indicates if the current position is the last in sequential processing
	bool IsLastSequencePos() const { return GetCurrentSequencePos() == (IsReverseSequense() ? 0 : GetMaxSequenceLength() - 1); }
	// Indicates if the network is working in recurrent mode
	bool IsRecurrentMode() const { return isRecurrentMode; }
	// Indicates that backpropagation was turned on for the current or the previous step
	bool IsBackwardPerformed() const { return isBackwardPerformed; }
	// Starts processing the sequence from the beginning
	void RestartSequence();
	// Enables or disables learning
	void DisableLearning();
	void EnableLearning();
	bool IsLearningEnabled() const { return isLearningEnabled; }
	// Checks and sets the auto-restart mode for each call to RunOnce/RunAndLearnOnce()
	bool GetAutoRestartMode() const { return autoRestartMode; }
	void SetAutoRestartMode(bool mode) { autoRestartMode = mode; }
	// Called by the layers to indicate that the layer should be reshaped before the next run
	// This may be necessary if the blob sizes change
	void RequestReshape(bool forcedReshape = false);
	// Called by the layers to indicate that the network must be rebuilt before the next run
	void ForceRebuild();
	// Checks if the network is going to be rebuilt before the next run
	// The method may be useful for controlling the rebuild frequency
	bool IsRebuildRequested() const { return isRebuildNeeded; }

	// Gets a reference to the random numbers generator
	CRandom& Random() { return random; }

	// Gets a reference to the math engine
	IMathEngine& GetMathEngine() const { return mathEngine; }

	// Accessing the optimizer
	CDnnSolver* GetSolver() { return solver; }
	const CDnnSolver* GetSolver() const { return solver; }
	void SetSolver(CDnnSolver* _solver);

	// Accessing the initializer. Xavier initialization is used by default
	CPtr<CDnnInitializer> GetInitializer() const { return initializer; }
	void SetInitializer(const CPtr<CDnnInitializer>& _initializer) { NeoAssert(_initializer != 0);  initializer = _initializer; }

	// Sets small enough values in layer parameters to zero
	void FilterLayersParams( float threshold );
	void FilterLayersParams( const CArray<const char*>& layerNames, float threshold );

	static const int ArchiveMinSupportedVersion = 1001;

	void Serialize( CArchive& archive );

	// Serializes network with data, required to resume training
	// When loading from checkpoint creates new solver (old pointers will point to an object, not used by this net anymore)
	void SerializeCheckpoint( CArchive& archive );

private:
	// Adds or deletes a layer
	void AddLayerImpl(CBaseLayer& layer) override;
	void DeleteLayerImpl(CBaseLayer& layer) override;

	CTextStream* log; // the logging stream
	int logFrequency;	// the logging frequency
	CPtr<CDnnSolver> solver;	// the layer parameter optimizer

	CRandom& random;	// the reference to the random numbers generator
	IMathEngine& mathEngine; // the reference to the math engine

	// The layer map
	CObjectArray<CBaseLayer> layers;
	CMap<CString, CBaseLayer*> layerMap;
	CArray<CBaseLayer*> sinkLayers;
	CArray<CBaseLayer*> sourceLayers;
	// The last run number
	int runNumber;
	// Indicates if the network needs rebuilding (the configuration has changed)
	bool isRebuildNeeded;
	// Indicates that backpropagation and learning should be performed on this step
	bool isBackwardPerformed;
	// Indicates that learning is enabled
	bool isLearningEnabled;
	// Indicates that the recurrent mode is on (for a sub-network of a recurrent layer)
	bool isRecurrentMode;

	// The initializer
	CPtr<CDnnInitializer> initializer;

	//////////////////////////////////////
	// For sequence processing
	int maxSequenceLength;
	int currentSequencePos;
	// Indicates that the sequence is processed in reverse order
	bool isReverseSequense;
	// The auto-restart mode for each RunOnce/RunAndLearnOnce() call
	bool autoRestartMode;
	// The low memory use mode
	bool isReuseMemoryMode;

	void setProcessingParams(bool isRecurrentMode, int sequenceLength, bool isReverseSequense, bool isBackwardPerformed);
	void runOnce(int curSequencePos);
	void backwardRunAndLearnOnce(int curSequencePos);
	void reshape();
	void rebuild();
	size_t getOutputBlobsSize() const;

	friend class CBaseLayer;
	friend class CCompositeLayer;
	friend class CRecurrentLayer;
};

inline CArchive& operator<<( CArchive& archive, const CDnn& dnn)
{
	const_cast<CDnn&>(dnn).Serialize( archive );
	return archive;
}

inline CArchive& operator>>( CArchive& archive, CDnn& dnn)
{
	dnn.Serialize( archive );
	return archive;
}

void NEOML_API SerializeLayer( CArchive& archive, IMathEngine& mathEngine, CPtr<CBaseLayer>& layer );

} // namespace NeoML

#include <NeoML/Dnn/Dnn.inl>
