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
#include <TestFixture.h>

using namespace NeoML;
using namespace NeoMLTest;

TEST( CDnnHeadTest, DISABLED_Simple )
{
    CRandom random( 42 );
    CDnn dnn( random, MathEngine() );

    CBaseLayer* x = Source( dnn, "SourceX" );
    x = FullyConnected( 512 )( x );
    x = Gelu()( x );

    CBaseLayer* y = Source( dnn, "SourceY" );
    y = FullyConnected( 512 )( y );
    y = Gelu()( y );

    CBaseLayer* z = Source( dnn, "SourceZ" );
    z = FullyConnected( 512 )( z );
    z = Gelu()( z );

    CDnnHead<
        CFullyConnectedLayer,
        CGELULayer,
        CFullyConnectedLayer,
        CReLULayer,
        CDropoutLayer,
        CFullyConnectedLayer
    > head(
        FullyConnected( 128 ),
        Gelu(),
        FullyConnected( 64 ),
        Relu(),
        Dropout( 0.5f ),
        FullyConnected( 1 )
    );

    // To be able run the same set of layers 3 times on 3 different inputs to get 3 different outputs links
    CBaseLayer* out = head( { x, y, z } );
    out = ConcatChannels()( CDnnLayerLink{ out, 0 }, CDnnLayerLink{ out, 1 }, CDnnLayerLink{ out, 3 } );

    CBaseLayer* labels = Source( dnn, "Labels" );
    BinaryCrossEntropyLoss()( out, labels );
}

