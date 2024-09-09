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

namespace NeoML {

inline float CBaseLayer::GetLearningRate() const
{
	return GetBaseLearningRate() * ( dnn == nullptr || dnn->owner == nullptr ? 1.f : dnn->owner->GetLearningRate() );
}

inline float CBaseLayer::GetL2RegularizationMult() const
{
	return GetBaseL2RegularizationMult()
		* ( dnn == nullptr || dnn->owner == nullptr ? 1.f : dnn->owner->GetL2RegularizationMult() );
}

inline float CBaseLayer::GetL1RegularizationMult() const
{
	return GetBaseL1RegularizationMult()
		* ( dnn == nullptr || dnn->owner == nullptr ? 1.f : dnn->owner->GetL1RegularizationMult() );
}

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

inline CString CBaseLayer::GetPath( const char* sep ) const
{
	return ( dnn == nullptr || dnn->owner == nullptr ) ? name : ( dnn->owner->GetPath( sep ) + sep + name );
}

inline void CBaseLayer::GetPath( CArray<CString>& path ) const
{
	path.DeleteAll();
	getPath( path );
}

inline void CBaseLayer::getPath( CArray<CString>& path ) const
{
	if( dnn == nullptr ) {
		return;
	}
	if( dnn->owner != nullptr ) {
		dnn->owner->getPath( path );
	}
	path.Add( name );
}

inline void CBaseLayer::CheckLayerArchitecture( bool expr, const char* message ) const
{
	if( !expr ) {
		CheckArchitecture( false, GetPath(), message );
	}
}

//---------------------------------------------------------------------------------------------------------------------

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

//---------------------------------------------------------------------------------------------------------------------

// Wrapper for the layer. Store layer type, init function and initialization params.
template<typename T>
class CLayerWrapper final {
public:
	CLayerWrapper( const char* prefix, CLambda<void( T* )> lambda );
	explicit CLayerWrapper( const char* prefix );
	// Be copied and moved by default

	// deprecated
	// Connects inputs to the layer and changes layer name.
	T* operator()( const char* name, const CDnnLayerLink& layer1,
		const CDnnLayerLink& layer2 = CDnnLayerLink(),
		const CDnnLayerLink& layer3 = CDnnLayerLink(),
		const CDnnLayerLink& layer4 = CDnnLayerLink(),
		const CDnnLayerLink& layer5 = CDnnLayerLink(),
		const CDnnLayerLink& layer6 = CDnnLayerLink() );
	// Connects inputs to the layer and changes layer name.
	T* operator()( const char* name, std::initializer_list<const CDnnLayerLink> layers );

	// deprecated
	// Connects inputs to the layer.
	T* operator()( const CDnnLayerLink& layer1,
		const CDnnLayerLink& layer2 = CDnnLayerLink(),
		const CDnnLayerLink& layer3 = CDnnLayerLink(),
		const CDnnLayerLink& layer4 = CDnnLayerLink(),
		const CDnnLayerLink& layer5 = CDnnLayerLink(),
		const CDnnLayerLink& layer6 = CDnnLayerLink() );
	// Connects inputs to the layer.
	T* operator()( std::initializer_list<const CDnnLayerLink> layers );

private:
	// Prefix for create layer name.
	const char* prefix;
	// Init function for new layer.
	CLambda<void( T* )> initFunc;
	// New layer.
	CPtr<T> layer;

	CString findFreeLayerName( const CDnn& first, const char* prefix ) const;
};

template<typename T>
CLayerWrapper<T>::CLayerWrapper( const char* _prefix, CLambda<void( T* )> _initFunc ) :
	prefix( _prefix ),
	initFunc( _initFunc )
{
}

template<typename T>
CLayerWrapper<T>::CLayerWrapper( const char* _prefix ) :
	prefix( _prefix )
{
	NeoAssert( prefix != 0 );
}

template<typename T>
T* CLayerWrapper<T>::operator()( const char* name,
	const CDnnLayerLink& layer1, const CDnnLayerLink& layer2,
	const CDnnLayerLink& layer3, const CDnnLayerLink& layer4,
	const CDnnLayerLink& layer5, const CDnnLayerLink& layer6 )
{
	return operator()( name, { layer1, layer2, layer3, layer4, layer5, layer6 } );
}

template<typename T>
T* CLayerWrapper<T>::operator()( const char* name, std::initializer_list<const CDnnLayerLink> layers )
{
	NeoAssert( layers.size() > 0 );
	NeoAssert( !layers.begin()->IsOptional() );
	NeoAssert( layers.begin()->IsValid() );
	NeoAssert( name != 0 );

	if( layer == 0 ) {
		CDnn* network = layers.begin()->Layer->GetDnn();
		NeoAssert( network != 0 );
		layer = new T( network->GetMathEngine() );
		if( !initFunc.IsEmpty() ) {
			initFunc( layer );
		}
		layer->SetName( name );
		network->AddLayer( *layer );
	}

	const int startIndex = layer->GetInputCount();
	for( int i = 0; i < static_cast<int>( layers.size() ); ++i ) {
		const CDnnLayerLink& inputLayer = *( layers.begin() + i );
		if( inputLayer.IsOptional() ) {
			break;
		}
		NeoAssert( inputLayer.IsValid() );
		NeoAssert( inputLayer.Layer->GetDnn() == layer->GetDnn() );
		layer->Connect( startIndex + i, *inputLayer.Layer, inputLayer.OutputNumber );
	}

	return layer;
}

template<typename T>
T* CLayerWrapper<T>::operator()(
	const CDnnLayerLink& layer1, const CDnnLayerLink& layer2,
	const CDnnLayerLink& layer3, const CDnnLayerLink& layer4,
	const CDnnLayerLink& layer5, const CDnnLayerLink& layer6 )
{
	return operator()( { layer1, layer2, layer3, layer4, layer5, layer6 } );
}

template<typename T>
T* CLayerWrapper<T>::operator()( std::initializer_list<const CDnnLayerLink> layers )
{
	NeoAssert( layers.size() > 0 );
	NeoAssert( layers.begin()->IsValid() );
	NeoAssert( layers.begin()->Layer->GetDnn() != nullptr );
	return operator()( findFreeLayerName( *layers.begin()->Layer->GetDnn(), prefix ), layers );
}

template<typename T>
CString CLayerWrapper<T>::findFreeLayerName( const CDnn& network, const char* prefix ) const
{
	const CString prefixStr( prefix );

	int index = 0;
	while( true ) {
		const CString newName = prefixStr + "_" + Str( index++ );
		if( !network.HasLayer( newName ) ) {
			return newName;
		}
	}
	NeoAssert( false );
	// make compiler happy
	return CString();
}

} // namespace NeoML
