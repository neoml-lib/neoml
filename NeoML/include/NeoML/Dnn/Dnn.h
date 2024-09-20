/* Copyright © 2017-2024 ABBYY

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
#include <NeoML/Dnn/DnnLambdaHolder.h>

// The macros for the internal name of a NeoML layer
// If this macros is used when declaring a class, that class may be registered as a NeoML layer
#define NEOML_DNN_LAYER( className ) friend class NeoML::CLayerClassRegistrar< className >;

// Registers the class as a NeoML layer
#define REGISTER_NEOML_LAYER( classType, name ) static NeoML::CLayerClassRegistrar< classType > __merge__1( _RegisterLayer, __LINE__ )( name, 0 );
#define REGISTER_NEOML_LAYER_EX( classType, name1, name2 ) static NeoML::CLayerClassRegistrar< classType > __merge__1( _RegisterLayer, __LINE__ )( name1, name2 );

//------------------------------------------------------------------------------------------------------------

namespace NeoML {

typedef CPtr<CBaseLayer> ( *TCreateLayerFunction )( IMathEngine& mathEngine );

void NEOML_API RegisterLayerClass( const char* className, const char* additionalName, const std::type_info& typeInfo, TCreateLayerFunction function );

void NEOML_API UnregisterLayerClass( const std::type_info& typeInfo );

bool NEOML_API IsRegisteredLayerClass( const char* className );

void NEOML_API GetRegisteredLayerClasses( CArray<const char*>& layerNames );

CPtr<CBaseLayer> NEOML_API CreateLayer( const char* name, IMathEngine& mathEngine );

CPtr<CBaseLayer> NEOML_API CreateLayer( const char* className, IMathEngine& mathEngine );

NEOML_API CString GetLayerClass( const CBaseLayer& layer );

template<class T>
CPtr<T> CreateLayer( const char* className, IMathEngine& mathEngine )
{
	return CheckCast<T>( CreateLayer( className, mathEngine ) );
}

//------------------------------------------------------------------------------------------------------------

template<class T>
class CLayerClassRegistrar {
public:
	CLayerClassRegistrar( const char* className, const char* additionalName );
	~CLayerClassRegistrar();

private:
	static CPtr<CBaseLayer> createObject( IMathEngine& mathEngine ) { return FINE_DEBUG_NEW T( mathEngine ); }
};

template<class T>
inline CLayerClassRegistrar<T>::CLayerClassRegistrar( const char* className, const char* additionalName )
{
	RegisterLayerClass( className, additionalName, typeid( T ), createObject );
}

template<class T>
inline CLayerClassRegistrar<T>::~CLayerClassRegistrar()
{
	UnregisterLayerClass( typeid( T ) );
}

//------------------------------------------------------------------------------------------------------------

// Forward declarations
class CDnn;
class CDnnLayerGraph;
class CBaseLayer;
class CCompositeLayer;
class CReferenceDnnFactory;
struct CReferenceDnnInfo;
struct CReferenceDnnInfoDeleter { void operator()( CReferenceDnnInfo* ); };
using TPtrOwnerReferenceDnnInfo = CPtrOwner<CReferenceDnnInfo, CReferenceDnnInfoDeleter>;

//------------------------------------------------------------------------------------------------------------

// The link between two layers, connecting one layer output to another layer input
struct CDnnLayerLink final {
	// the pointer to the linked layer
	CBaseLayer* Layer;
	// the number of the output to which the connection leads
	int OutputNumber = -1;

	// Default value for optional inputs.
	CDnnLayerLink() = default;
	// Be copied and moved by default

	// Converting constructor
	CDnnLayerLink( CBaseLayer* layer, int outputNumber = 0 ) :
		Layer( layer ),
		OutputNumber( outputNumber )
	{
		NeoAssert( Layer != nullptr );
		NeoAssert( OutputNumber >= 0 );
	}

	// Is this layer optional, i.e. created by CLayerOutout() default constructor.
	bool IsOptional() const { return Layer == nullptr && OutputNumber == -1; }
	// Is the layer output valid?
	bool IsValid() const { return Layer != nullptr && OutputNumber >= 0; }
};

//------------------------------------------------------------------------------------------------------------

// CBaseLayer is the base class for all layers with which the network can function. 
// Each layer has a string name that should be unique in the network. Each layer may have 
// one or several inputs and one or several outputs.
class NEOML_API CBaseLayer : public virtual IObject {
public:
	CBaseLayer( IMathEngine& mathEngine, const char* name, bool isLearnable );

	// Retrieves the reference to the IMathEngine with which the layer was created
	IMathEngine& MathEngine() const { return mathEngine; }

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

	// Gets the path to this Layer
	// If this layer directly belongs to the root CDnn then this path consists of the name of this layer only
	// Otherwise it contains the names of all the composites from the root till this layer separated by '/'
	//
	// e.g. layer "InputHidden" inside of CLstmLayer named "LSTM", which is inside of CCompositeLayer named "Encoder"
	// has path "Encoder/LSTM/InputHidden"
	CString GetPath( const char* sep = "/" ) const;
	// Path in form suitable for dnn->GetLayer( CArray<CString>& path );
	// Returns an empty array if the path cannot be constructed.
	void GetPath( CArray<CString>& path ) const;

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

	// Product of the layer's coefficient with all it's owners' ones (when layer is inside of composite/recurrent)
	float GetLearningRate() const;
	float GetL2RegularizationMult() const;
	float GetL1RegularizationMult() const;

	// Begins processing a new sequence
	// The method is overloaded for the composite layer and the backward link layer
	virtual void RestartSequence() {} 

	void Serialize(CArchive& archive) override;

	// Indicates that backpropagation should be performed for the layer 
	// even if there are no trainable layers before it
	bool GetBackwardForced() const { return isBackwardForced; }
	void SetBackwardForced(bool forced);

	// Returns the total size of all output blobs together
	virtual size_t GetOutputBlobsSize() const;

	// Releases all temporary resources allocated for the layer
	virtual void CleanUp( bool totalCleanUp = false );

	// Returns the number of trainable parameters (floats or ints) in all of this layer's parameters blobs
	// Returns the number of trainable parameters of its internal layers, if layer is composite or recurrent
	virtual size_t GetTrainableParametersSize() const;

	// Enable profile timer for RunOnce
	virtual void EnableProfile( bool profile ) { useTimer = profile; }
	// Returns number of RunOnce calls since last Reshape
	int GetRunOnceCount() const { return runOnceCount; }
	// Returns total time of RunOnce calls (in milliseconds) since last Reshape
	IPerformanceCounters::CCounter::TCounterType GetRunOnceTime() const { return runOnceTime / 1000000; }

protected:
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
	// Layer may contain null paramBlob of given index, specialization for transferParamsBlob
	virtual bool ContainsNullParamBlob( int ) const { return false; }
	// Special case, specialization for transferParamsBlob
	virtual bool IsLearnableWithEmptyParamBlobs() const { return false; }
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

	// Initializes the parameters blob using the specified initializing algorithm
	// If inputSize == 0, the blob will have the (inputBlobs[input] / 2) size
	// (halving the matrix size means that we assume about half the neurons will be 0)
	void InitializeParamBlob( int input, CDnnBlob& blob, int inputSize = 0 );
	// Called by a layer to warn that Reshape should be done before the next run
	void ForceReshape();

	virtual void OnDnnChanged( CDnn* ) {}

	// Fills with zeros the parameters that are less (but not equal) than a given threshold
	virtual void FilterLayerParams( float /*threshold*/ ) {}

	// Allocates the output blobs
	// The default implementation creates the outputBlobs array using the output descriptions
	virtual void AllocateOutputBlobs();

	// The following section contains interface for the memory optimization during training
	// The key idea is that the layer may provide additional information about blobs required
	// for backward and for learning

	// Blob types which are used by layer during backward and learn
	static const int TInputBlobs = 1 << 0;
	static const int TOutputBlobs = 1 << 1;

	// The following methods are called during reshape stage and may use anything available in Reshape methods
	// Blob types required for the correct work of BackwardOnce
	virtual int BlobsForBackward() const { return TInputBlobs | TOutputBlobs; }
	// Blob types required for the correct work of LearnOnce
	virtual int BlobsForLearn() const { return TInputBlobs | TOutputBlobs; }

	// Indicates if the layer overwrite its inputs
	bool InputsMayBeOverwritten() const;
	// Enables in-place process
	// This flag must be set from Reshape method
	// It is reset to false right each time before the Reshape call
	void EnableInPlace( bool enable ) { isInPlace = enable; }
	bool IsInPlace() const { return isInPlace; }

	// Throw check exception if expr is false
	void CheckLayerArchitecture( bool expr, const char* message ) const;

private:
	// Describes an input connection
	struct CInputInfo final {
		CString Name; // the name of the layer that is connected to the input
		int OutputNumber = NotFound; // the number of that layer's output that is connected to the input
	};
	// Indicates if back propagation should be performed for the layer
	enum TBackwardStatus : char {
		BS_Unknown,
		BS_NeedsBackward,
		BS_DoesntNeedBackward
	};
	// The temporary blob cache for sequence processing in a recurrent layer
	enum TBlobCacheType {
		BCT_Input,
		BCT_Output,
		BCT_Runtime,

		BCT_Count
	};

	// Input layer links
	CArray<CDnnLayerLink> inputLinks;
	// The number of connections to each layer output
	CArray<int> outputs;
	// The last layer which uses this outputs
	CArray<const CBaseLayer*> lastOutputUser;

	// The number of output diffs ready for backpropagation
	// When the ready diffs and the outputs numbers become the same, the layer is ready for backpropagation
	// (BackwardRunAndLearnOnce method may be called for this layer)
	CArray<int> readyOutputDiffs;

	// The temporary data for backpropagation or learning
	CObjectArray<CDnnBlob> runtimeBlobs;
	CArray<CPtr<CDnnBlob>*> runtimeBlobPtrs;

	CObjectArray<CDnnBlob> blobCache[BCT_Count];

	CArray<CInputInfo> inputs; // inputs list

	IMathEngine& mathEngine;   // the layer's MathEngine
	CString name;              // the layer name

	// Owner network; may be null if the layer does not belong to a network
	CDnn* dnn = nullptr;
	// The total time of RunOnce calls since last Reshape in nanoseconds
	IPerformanceCounters::CCounter::TCounterType runOnceTime = 0;

	// Fields used for memory optimization during training
	int allocatedBlobs; // the mask of currently allocated blobs
	int blobsNeededForBackward; // the mask of blobs needed for backward and learn
	// The number of the last network run. As a layer may be called several times 
	// during RunOnce method execution, it will first check the run number 
	// and do no calculations if it is still the same run
	int lastRunNumber = 0;
	// The number of graphs with which the layer is connected
	int graphCount = 0;
	// The total number of RunOnce calls since last Reshape
	int runOnceCount = 0;

	// The base learning rate (may vary inside the network depending on the learning strategy)
	float baseLearningRate = 1;
	// Base regularization multiplier (may vary inside the network depending on the learning strategy)
	float baseL2RegularizationMult = 1;
	float baseL1RegularizationMult = 1;

	// Indicates if the layer may be trained
	const bool isLearnable;
	// Indicates if learning is enabled for the layer
	bool isLearningEnabled = true;
	// Indicates if back propagation should be performed for the layer
	TBackwardStatus isBackwardNeeded = BS_Unknown;
	// Forces back propagation
	bool isBackwardForced = false;
	// Forces reshaping the layer even with unchanged inputs
	// May be useful if you change the parameters that determine the output size
	bool forcedReshape = true;
	// Indicates if the layer should be reshaped
	bool isReshapeNeeded = true;
	// Indicates if the layer performs in-place processing (after the Reshape method call)
	bool isInPlace = false;
	// Use timer to calculate run once time and hit count
	bool useTimer = false;

	// Set the 'dist' layer's paramBlobs to point to the data of this layer's paramBlobs
	void transferParamsBlob(CBaseLayer& dist) const;
	// Technical method for recursion in GetPath( CArray<CString>& path )
	void getPath( CArray<CString>& path ) const;
	void sequentialModeIfRecurrent();
	void nonSequentialModeIfRecurrent();
	// Switches the specified blobs into sequence processing mode
	void switchBlobsToSequentialMode(CObjectArray<CDnnBlob>& blobs, TBlobCacheType cacheType, bool storeParent);
	void switchBlobsToNonSequentialMode(CObjectArray<CDnnBlob>& blobs, TBlobCacheType cacheType, bool clear);
	void clearAllRuntimeBlobs();
	// Clones a blob to store diffs
	CDnnBlob* cloneBlobForDiff(const CBlobDesc& desc);
	// Indicates if the layer is composite (contains another sub-network)
	virtual bool isComposite() const { return false; }
	// Sets the mask of allocated blobs
	// If some some blobs are not marked as allocated, they will be freed during this call
	void setAllocatedBlobs( int newMask );

	//------------------------------------------------------
	// The methods and data for interacting with the network

	void setDnn( CDnn* newDnn );
	void link();
	void addOutput(int number);
	void unlink();
	void cleanUp( bool total, bool linked );
	void buildOrder();
	void reshape();
	void setInputDesc(int i);
	void runOnce();
	void recheckBackwardNeeded();
	void backwardRunAndLearnOnce();
	void transferDiffBlob( CDnnBlob* diffBlob, int outputNum );

	friend class CDnn;
	friend class CDnnLayerGraph;
	friend class CDnnSolver;
	friend class CCompositeLayer;
	friend class CReferenceDnnFactory;
};

//------------------------------------------------------------------------------------------------------------

class CActivationDesc;
// Common interface for all activation functions
class NEOML_API IActivationLayer {
public:
	virtual ~IActivationLayer() = default;
	// Get the name and settings of the activation
	virtual CActivationDesc GetDesc() const = 0;
};

//------------------------------------------------------------------------------------------------------------

// CDnnLayerGraph is the base class for a layer graph
class NEOML_API CDnnLayerGraph {
public:
	virtual ~CDnnLayerGraph() = default;

	// Accessing the layers
	virtual int GetLayerCount() const = 0;
	virtual void GetLayerList( CArray<char const*>& layerList ) const = 0;
	virtual CPtr<CBaseLayer> GetLayer( const char* name ) = 0;
	virtual CPtr<const CBaseLayer> GetLayer( const char* name ) const = 0;
	virtual CPtr<CBaseLayer> GetLayer( const CArray<CString>& path ) = 0;
	virtual CPtr<const CBaseLayer> GetLayer( const CArray<CString>& path ) const = 0;
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

//------------------------------------------------------------------------------------------------------------

// Gets the NeoML exception handler for math engine
NEOML_API IMathEngineExceptionHandler* GetExceptionHandler();

// deprecated
void NEOML_API EnableSingleThreadMode( bool enable );
// deprecated
bool NEOML_API IsSingleThreadModeOn();

// Gets a math engine that performs calculations on CPU.
// The engine does not need to be destroyed.
NEOML_API IMathEngine& GetDefaultCpuMathEngine();

// deprecated
NEOML_API IMathEngine& GetSingleThreadCpuMathEngine();
// deprecated
NEOML_API IMathEngine& GetMultiThreadCpuMathEngine();

// Gets a math engine that works on the recommended GPU
// Set memoryLimit = 0 to use all available memory
// Returns 0 if the GPU is not available
// The engine SHOULD be destroyed after use with standart delete
NEOML_API IMathEngine* GetRecommendedGpuMathEngine( size_t memoryLimit );

//------------------------------------------------------------------------------------------------------------

// CDnn class represents a neural network
class NEOML_API CDnn : public CDnnLayerGraph {
public:
	CDnn( CRandom& random, IMathEngine& mathEngine, const CCompositeLayer* owner = nullptr );
	~CDnn() override;

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
	CPtr<CBaseLayer> GetLayer(const CArray<CString>& path) override;
	CPtr<const CBaseLayer> GetLayer(const CArray<CString>& path) const override;
	bool HasLayer( const char* name ) const override { return layerMap.Has( name ); }

	// Runs the network: all data from the input blobs is used
	void RunOnce();
	// Runs the network and performs a backward pass with the input data
	void RunAndBackwardOnce();
	// Runs the network, performs a backward pass and updates the trainable weights
	void RunAndLearnOnce();

	// Releases all temporary resources allocated for RunAndBackwardOnce()
	void CleanUp( bool totalCleanUp = false );

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
	// Shares its weights with other reference dnns
	bool IsReferenceDnn() const { return ( getOwnerDnn().referenceDnnInfo != nullptr ); }

	// Gets a reference to the random numbers generator
	CRandom& Random() { return random; }
	const CRandom& Random() const { return random; }

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

	// Enables profiling for all the layers in the network
	void EnableProfile( bool profile );

private:
	// The layer map
	CMap<CString, CBaseLayer*> layerMap;
	CObjectArray<CBaseLayer> layers;
	CArray<CBaseLayer*> sinkLayers;
	CArray<CBaseLayer*> sourceLayers;

	CRandom& random; // the reference to the random numbers generator
	IMathEngine& mathEngine; // the reference to the math engine

	// The layer parameter optimizer
	CPtr<CDnnSolver> solver;
	// The initializer
	CPtr<CDnnInitializer> initializer;
	// Reference information
	TPtrOwnerReferenceDnnInfo referenceDnnInfo;

	const CBaseLayer* owner = nullptr; // the composite containing this CDnn (if exists)
	CTextStream* log = nullptr; // the logging stream

	// The logging frequency
	int logFrequency = 100;
	// The last run number
	int runNumber = -1;

	//////////////////////////////////////
	// For sequence processing
	int maxSequenceLength = 1;
	int currentSequencePos = 0;
	// Indicates that the sequence is processed in reverse order
	bool isReverseSequense = false;
	// The auto-restart mode for each RunOnce/RunAndLearnOnce() call
	bool autoRestartMode = true;

	// The low memory use mode
	bool isReuseMemoryMode = false;
	// Indicates if the network needs rebuilding (the configuration has changed)
	bool isRebuildNeeded = false;
	// Indicates that backpropagation and learning should be performed on this step
	bool isBackwardPerformed = false;
	// Indicates that learning is enabled
	bool isLearningEnabled = true;
	// Indicates that the recurrent mode is on (for a sub-network of a recurrent layer)
	bool isRecurrentMode = false;

	// Adds or deletes a layer
	void AddLayerImpl( CBaseLayer& layer ) override;
	void DeleteLayerImpl( CBaseLayer& layer ) final;
	// Should be called in all internals methods
	CBaseLayer* getLayer( const char* name );
	// Should be called in all internals methods
	CBaseLayer* getLayer( const CArray<CString>& path );

	void setProcessingParams(bool isRecurrentMode, int sequenceLength, bool isReverseSequense, bool isBackwardPerformed);
	void runOnce(int curSequencePos);
	void backwardRunAndLearnOnce(int curSequencePos);
	void reshape();
	void rebuild();
	size_t getOutputBlobsSize() const;
	const CDnn& getOwnerDnn() const
		{ return ( owner == nullptr || owner->GetDnn() == nullptr ) ? *this : owner->GetDnn()->getOwnerDnn(); }

	friend class CBaseLayer;
	friend class CCompositeLayer;
	friend class CRecurrentLayer;
	friend class CReferenceDnnFactory;
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

//------------------------------------------------------------------------------------------------------------

// Result of CReferenceDnnFactory::CreateReferenceDnn
// NOTE: Class CDnnReference should be created using CPtr only.
class NEOML_API CDnnReference : public IObject {
public:
	CDnn Dnn;

protected:
	CDnnReference( CRandom& random, IMathEngine& mathEngine ) : Dnn( random, mathEngine ) {}
	// Use CPtr<CDnnReference> to create the class
	~CDnnReference() override = default;

	friend class CReferenceDnnFactory;
};

// This class can initialize a reference dnn, that has the same configuration as the original dnn
// and shares parameter blobs with the original dnn to save memory.
// Useful for multi-threaded inference where each thread can operate own reference dnn independently.
// Learning is disabled for both the original dnn and the reference dnn.
// Creates a copy of the original dnn's random generator to use it for inference.
// NOTE: Class CReferenceDnnFactory should be created using CPtr only.
class NEOML_API CReferenceDnnFactory : public IObject {
public:
	// Archive should contain trained dnn, ready for inference
	// NOTE: mathEngine should be CPU only and live longer than CReferenceDnnFactory
	CReferenceDnnFactory( IMathEngine& mathEngine, CArchive& archive, int seed = 42, bool optimizeDnn = true );
	// Dnn should be trained and ready for inference,
	// Dnn will be copied for internal storage to be non-disturbed, one can use argument dnn further as one wants
	// NOTE: mathEngine should be CPU only and live longer than CReferenceDnnFactory
	CReferenceDnnFactory( IMathEngine& mathEngine, const CDnn& dnn, bool optimizeDnn = true );
	// Dnn should be trained and ready for inference, it will be moved inside and cannot be used outside.
	// NOTE: mathEngine should be CPU only and live longer than CReferenceDnnFactory
	CReferenceDnnFactory( CDnn&& dnn, bool optimizeDnn = true );

	// Thread-safe coping of originalDnn, increments the counter
	// NOTE: The original dnn used to copy reference dnns may be also used as one more reference dnn (optimization)
	//       The 'getOriginDnn' flag must be used strictly for the only thread.
	CPtr<CDnnReference> CreateReferenceDnn( bool getOriginDnn = false );

protected:
	// Use CPtr<CReferenceDnnFactory> to create the class
	~CReferenceDnnFactory() override = default;

private:
	CPtr<CDnnReference> Origin; // The dnn to make reference dnns

	// Technical constructor
	CReferenceDnnFactory( CRandom random, IMathEngine& mathEngine );

	// Internal method of loading to the origin dnn
	void serialize( CArchive& archive, bool optimizeDnn );
	// Thread-safe coping the state (with no copy paramBlobs the pointers used) of a dnn to a new dnn
	void initializeReferenceDnn( CDnn& dnn, CDnn& newDnn, TPtrOwnerReferenceDnnInfo&& info );
	// Update layers' settings for a better paramBlobs sharing
	static void allowLayersToShareParamBlobs( CDnn& dnn );
};

} // namespace NeoML

//------------------------------------------------------------------------------------------------------------

#include <NeoML/Dnn/Dnn.inl>

