/* Copyright © 2021-2024 ABBYY

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

#include <atomic>
#include <thread>
#include <TestFixture.h>

using namespace NeoML;
using namespace NeoMLTest;

TEST( CDnnBlobTest, InitWindowBlob )
{
    MathEngine().CleanUp();
    {
        CPtr<CDnnBlob> parent = CDnnBlob::CreateDataBlob( MathEngine(), CT_Float, 16, 1, 1 );
        CPtr<CDnnBlob> window = CDnnBlob::CreateWindowBlob( parent );

        EXPECT_TRUE( window->GetData().IsNull() == false );
    }
    EXPECT_TRUE( MathEngine().GetPeakMemoryUsage() == 16 * sizeof( float ) );
    EXPECT_TRUE( MathEngine().GetCurrentMemoryUsage() == 0 );
}

TEST( CDnnBlobTest, BufferTest )
{
    CPtr<CDnnBlob> blob = CDnnBlob::CreateDataBlob( MathEngine(), CT_Float, 16, 1, 1 );
    EXPECT_FALSE( blob->GetData().IsNull() );

    CDnnBlobBuffer<float> buffer( *blob, TDnnBlobBufferAccess::Write );
    EXPECT_NE( nullptr, buffer.Ptr() );
    ::memset( buffer, 0, buffer.Size() * sizeof( float ) );

    EXPECT_FALSE( buffer.IsClosed() );
    buffer.Close();
    EXPECT_TRUE( buffer.IsClosed() );
}

//---------------------------------------------------------------------------------------------------------------------

TEST( CDnnBlobTest, BufferMemoryThresholdTest )
{
    auto testMethod = []( size_t threshold, bool init, size_t &sumMemoryInPools )
    {
        if( init ) {
            MathEngine().CleanUp();
            MathEngine().SetReuseMemoryMode( true );
            MathEngine().SetThreadBufferMemoryThreshold( threshold );
        }

        const size_t peakMemory = MathEngine().GetCurrentMemoryUsage();
        const size_t reusedMemory = ( init ? 0 : threshold );
        {
            CPtr<CDnnBlob> blob1 = CDnnBlob::CreateDataBlob( MathEngine(), CT_Float, int(threshold / sizeof( float )), 1, 1 );
            ASSERT_TRUE( blob1 != nullptr && !blob1->GetData().IsNull() );
            CPtr<CDnnBlob> blob2 = CDnnBlob::CreateDataBlob( MathEngine(), CT_Float, int( threshold / sizeof( float ) + 1), 1, 1 );
            ASSERT_TRUE( blob2 != nullptr && !blob2->GetData().IsNull() );
            EXPECT_EQ( MathEngine().GetCurrentMemoryUsage(), peakMemory + threshold - reusedMemory + threshold + sizeof( float ) );
        }
        const size_t memoryInPools = MathEngine().GetMemoryInPools() - reusedMemory;
        EXPECT_EQ( memoryInPools, threshold - reusedMemory );

        EXPECT_EQ( MathEngine().GetCurrentMemoryUsage() - memoryInPools, peakMemory );
        sumMemoryInPools += memoryInPools;
    };

    size_t sumMemoryInPools = 0;
    {
        testMethod( 256, /*init*/true, sumMemoryInPools );

        std::thread thread( testMethod, 512, /*init*/true, std::ref( sumMemoryInPools ) );
        thread.join();

        testMethod( 256, /*init*/false, sumMemoryInPools );
    }
    EXPECT_EQ( sumMemoryInPools, 256 + 512 );
}

//---------------------------------------------------------------------------------------------------------------------

#if FINE_PLATFORM( FINE_WINDOWS ) || !defined( NEOML_USE_FINEOBJ )

namespace NeoMLTest {

enum class TTransferType {
    PoolToPool, PoolToHeap, HeapToPool, HeapToHeap
};

class CDnnBlobTest : public CNeoMLTestFixture, public ::testing::WithParamInterface<TTransferType> {
public:
	static bool InitTestFixture() { return true; }
	static void DeinitTestFixture() {}
};

static void testTransferBlobInThreadsImpl( TTransferType type )
{
    const TMathEngineType met = MathEngine().GetType();
    switch( met ) {
        case MET_Cpu:
            break;
        case MET_Cuda:
            if( TTransferType::PoolToPool == type ) {
                break;
            }
            GTEST_LOG_( INFO ) << "Skipped (" << int( type ) << ") for met=" << met;
            return;
        case MET_Metal:
        case MET_Vulkan:
            GTEST_LOG_( INFO ) << "Skipped (" << int(type) << ") for met=" << met;
            return;
        default:
            EXPECT_TRUE( false );
    }

    DeleteMathEngine();
    IMathEngine& mathEngine = MathEngine(); // create unique MathEngine

    std::atomic<bool> created{ false };
    std::atomic<bool> transfered{ false };
    std::atomic<bool> cleaned{ false };

    const int blobSize = 1000; // bytes
    const int blobBufferSize = 1024; // bytes
    const int blobCheckSize = 200; // bytes
    const int blobCheckBufferSize = 256; // bytes
    const int blobTransferedSize = 500; // bytes
    const int blobTransferedBufferSize = 512; // bytes

    CPtr<CDnnBlob> blobTransfer;

    std::thread oldThread( [&]()
    {
        EXPECT_TRUE( mathEngine.GetPeakMemoryUsage() == 0 ); // no allocated memory
        const bool usePools = ( type == TTransferType::PoolToPool || type == TTransferType::PoolToHeap );
        mathEngine.SetReuseMemoryMode( usePools );
        EXPECT_TRUE( mathEngine.GetMemoryInPools() == 0 ); // no allocated non-used memory

        {
            const int blobCount = blobSize / sizeof( float );
            const int blobCheckCount = blobCheckSize / sizeof( float );
            const int blobTransferedCount = blobTransferedSize / sizeof( float );

            // Creating blobs
            CPtr<CDnnBlob> blob = CDnnBlob::CreateDataBlob( mathEngine, CT_Float, blobCount, 1, 1 );
            CPtr<CDnnBlob> blobCheck = CDnnBlob::CreateDataBlob( mathEngine, CT_Float, blobCheckCount, 1, 1 );
            blobTransfer = CDnnBlob::CreateDataBlob( mathEngine, CT_Float, blobTransferedCount, 1, 1 );

            // Allocated memory of 3 blobs on OLD thread
            EXPECT_TRUE( mathEngine.GetPeakMemoryUsage() == ( usePools
                ? ( blobTransferedBufferSize + blobCheckBufferSize + blobBufferSize )
                : ( blobTransferedSize + blobCheckSize + blobSize ) ) );
            // All allocated memory is busy, no non-used memory
            EXPECT_TRUE( mathEngine.GetMemoryInPools() == 0 );
            ( void ) created.exchange( true );

            while( !transfered ); // wait
            // Now OLD thread contains memory of only non-trasfered blobs after transfer
            blob.Release(); // Destroy blob

            // Memory still in pool after the blob's destroyed, if pools used
            EXPECT_TRUE( mathEngine.GetMemoryInPools() == ( usePools ? blobBufferSize : 0 ) ); // allocated non-used memory
            // Clean-up non-used memory for this OLD thread
            mathEngine.CleanUp();
            EXPECT_TRUE( mathEngine.GetMemoryInPools() == 0 );
        }  // Destroy blobCheck

        EXPECT_TRUE( mathEngine.GetMemoryInPools() == ( usePools ? blobCheckBufferSize : 0 ) ); // allocated non-used memory
        // Finally clean-up all non-used memory for this OLD thread
        mathEngine.CleanUp();
        EXPECT_TRUE( mathEngine.GetMemoryInPools() == 0 );
        
        ( void ) cleaned.exchange( true );
    } );

    std::thread newThread( [&]()
    {
        const bool useOldPools = ( type == TTransferType::PoolToPool || type == TTransferType::PoolToHeap );
        const bool usePools = ( type == TTransferType::PoolToPool || type == TTransferType::HeapToPool );
        mathEngine.SetReuseMemoryMode( usePools );
        EXPECT_TRUE( mathEngine.GetMemoryInPools() == 0 ); // no non-used allocated memory

        while( !created ); // wait

        // Allocated memory of 3 blobs on OLD thread
        EXPECT_TRUE( mathEngine.GetPeakMemoryUsage() == ( useOldPools
            ? ( blobTransferedBufferSize + blobCheckBufferSize + blobBufferSize )
            : ( blobTransferedSize + blobCheckSize + blobSize ) ) );

        CPtr<CDnnBlob> blobTransfered = new CDnnBlob( mathEngine ); // create empty blob
        *blobTransfered = std::move( *blobTransfer ); // TransferDataToThisThread()
        CPtr<CDnnBlob> blobTransferedAgain = new CDnnBlob( std::move( *blobTransfered ) ); // move again
        // Now NEW thread contains memory of only one trasfered blob
        blobTransfer.Release();
        blobTransfered.Release();

        // All allocated memory is busy, no non-used memory
        EXPECT_TRUE( mathEngine.GetMemoryInPools() == 0 );
        ( void ) transfered.exchange( true );
        while( !cleaned ); // wait

        blobTransferedAgain->Fill( 2 ); // OK!
        blobTransferedAgain.Release(); // Destroy blob
        //blobCheck->Fill( 0 ); // Error! segfault

        // Memory still in pool after the blob's destroyed, if pools used
        EXPECT_TRUE( mathEngine.GetMemoryInPools() ==
            ( ( type == TTransferType::PoolToPool ) ? blobTransferedBufferSize : 0 ) );

        // Finally clean-up all non-used memory for this NEW thread
        mathEngine.CleanUp();
        EXPECT_TRUE( mathEngine.GetMemoryInPools() == 0 );
        //blobTransfer->Fill( 1 ); // Error! segfault
    } );

    oldThread.join();
    newThread.join();

    EXPECT_TRUE( mathEngine.GetPeakMemoryUsage() > 0 );
    mathEngine.ResetPeakMemoryUsage();
    EXPECT_TRUE( mathEngine.GetPeakMemoryUsage() == 0 );
}

} // namespace NeoMLTest

TEST_P( CDnnBlobTest, TransferBlobInThreads )
{
    TTransferType type = GetParam();
    testTransferBlobInThreadsImpl( type );
}

INSTANTIATE_TEST_CASE_P( CDnnBlobTestInstantiation, CDnnBlobTest,
    ::testing::Values(
        TTransferType::PoolToPool, TTransferType::PoolToHeap, TTransferType::HeapToPool, TTransferType::HeapToHeap
    )
);

#endif // FINE_WINDOWS || !NEOML_USE_FINEOBJ
