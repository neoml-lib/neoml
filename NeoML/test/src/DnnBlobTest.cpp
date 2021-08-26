/* Copyright © 2021 ABBYY Production LLC

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

#include <TestFixture.h>

TEST(CDnnBlobTest, InitWindowBlob)
{
    NeoML::IMathEngine& engine = NeoML::GetSingleThreadCpuMathEngine();
    CPtr<NeoML::CDnnBlob> parent = NeoML::CDnnBlob::CreateDataBlob( engine, NeoML::CT_Float, 16, 1, 1 );
    CPtr<NeoML::CDnnBlob> blob = NeoML::CDnnBlob::CreateWindowBlob( parent );

    ASSERT_FALSE( blob->GetData().IsNull() );
}

TEST(CDnnBlobTest, BufferTest)
{
    NeoML::IMathEngine& engine = NeoML::GetSingleThreadCpuMathEngine();
    CPtr<NeoML::CDnnBlob> blob = NeoML::CDnnBlob::CreateDataBlob( engine, NeoML::CT_Float, 16, 1, 1 );
    ASSERT_FALSE( blob->GetData().IsNull() );

    NeoML::CDnnBlobBuffer<float> buffer( *blob, 0, blob->GetDataSize(), NeoML::TDnnBlobBufferAccess::Write );
    ASSERT_NE( nullptr, buffer.Ptr() );
    ::memset( buffer, 0, buffer.Size() * sizeof(float) );

    EXPECT_FALSE( buffer.IsClosed() );
    buffer.Close();
    EXPECT_TRUE( buffer.IsClosed() );
}
