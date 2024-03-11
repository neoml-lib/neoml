/* Copyright © 2024 ABBYY

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

#include <NeoML/Dnn/DnnHead.h>
#include <NeoML/Dnn/Layers/DnnHeadAdapterLayer.h>
#include <TestFixture.h>

using namespace NeoML;
using namespace NeoMLTest;

TEST( CDnnHeadTest, DISABLED_Simple ) // TODO: ??? remove "DISABLED_" from name
{
    IMathEngine& mathEngine = MathEngine();
    CRandom random( 42 );
    CDnn dnn( random, mathEngine );

    // To be able run the same set of layers 3 times
    // on 3 different inputs to get 3 different outputs links
    std::shared_ptr<CDnn> head = CDnnHead<
        CFullyConnectedLayer,
        CGELULayer,
        CFullyConnectedLayer,
        CReLULayer,
        CDropoutLayer,
        CFullyConnectedLayer
    >(
        random, mathEngine,
        FullyConnected( 128 ),
        Gelu(),
        FullyConnected( 64 ),
        Relu(),
        Dropout( 0.5f ),
        FullyConnected( 1 )
    );

    CBaseLayer* x = Source( dnn, "srcX" );
    x = FullyConnected( 512 )( x );
    x = Gelu()( x );
    x = DnnHeadAdapter( head )( x ); // #1 head dnn call

    CBaseLayer* y = Source( dnn, "srcY" );
    y = FullyConnected( 512 )( y );
    y = Gelu()( y );
    y = DnnHeadAdapter( head )( y ); // #2 head dnn call

    CBaseLayer* z = Source( dnn, "srcZ" );
    z = FullyConnected( 512 )( z );
    z = Gelu()( z );
    z = DnnHeadAdapter( head )( z ); // #3 head dnn call

    CBaseLayer* out = ConcatChannels()( x, y, z );

    CBaseLayer* labels = Source( dnn, "labels" );
    BinaryCrossEntropyLoss()( out, labels );

    // TODO: ???
    dnn.RunOnce();
}

