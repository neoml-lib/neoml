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

#include <map>
#include <chrono>

namespace NeoML {

#ifdef JIT_DEBUG
static bool printTimers = getenv( "PRINT_TIMERS" ) != nullptr;
#endif

class CJitDebug {
public:
    struct CKey{
        int filterCount;
        int channelCount;
        int filterHeight;
        int filterWidth;
        int paddingHeight;
        int paddingWidth;
        int strideHeight;
        int strideWidth;
        int dilationHeight;
        int dilationWidth;
        int resultHeight;
        int resultWidth;

        bool operator==( const CKey& lhs ) const {
            return memcmp( this, &lhs, sizeof( CKey ) ) == 0;
        }

        struct Hasher
        {
            size_t operator()( const CKey& key ) const {
                size_t hash = 0;
                const size_t* keyPtr = reinterpret_cast<const size_t*>( &key );
                for( int i = 0; i < sizeof( key ) / sizeof( size_t ); i++ ) {
                    hash ^= keyPtr[i];
                }
                return hash;
            }
        };


    };

    struct CResult{
        CResult() : count( 0 ), reuseCount( 0 ), prepareMs( 0.0f ), processMs( 0.0f ), codeSize( 0 ) {}
        size_t count;
        size_t reuseCount;
        std::chrono::duration<float> prepareMs;
        std::chrono::duration<float> processMs;
        size_t codeSize;
    };
public:
#ifndef JIT_DEBUG
    CJitDebug( ... ) = default;
    ~CJitDebug() = default;
    void StartPrepare() {}
    void StopPrepare() {}
    void StartProcess() {}
    void StopProcess() {}
    void SetCodeSize( size_t ) {}
    void PrintResult() {}
# else
    CJitDebug( int filterCount, int channelCount, int filterHeight, int filterWidth,
               int paddingHeight, int paddingWidth, int strideHeight, int strideWidth,
               int dilationHeight, int dilationWidth, int resultHeight, int resultWidth
               ) : key{ filterCount, channelCount, filterHeight, filterWidth,
                                    paddingHeight, paddingWidth, strideHeight, strideWidth,
                                    dilationHeight, dilationWidth, resultHeight, resultWidth },
                               prepareMs( 0.0f ), processMs( 0.0f ), codeSize( 0 ) {}
    ~CJitDebug() {
        auto& inst = res[key];
        inst.count++;
        inst.prepareMs += prepareMs;
        inst.processMs += processMs;
        inst.codeSize += codeSize;
        if( codeSize == 0 ) {
            inst.reuseCount++;
        }
    }
    void StartPrepare() {
        prepareMsStart = Time::now();
    }
    void StopPrepare() {
        prepareMs += Time::now() - prepareMsStart;
    }
    void StartProcess() {
        processMsStart = Time::now();
    }
    void StopProcess() {
        processMs += Time::now() - processMsStart;
    }
    void SetCodeSize( size_t _codeSize ) {
        codeSize = _codeSize;
    }
    static void PrintResult() {
        if( printTimers ) {
            printf( "FC;C;FW;FH;DW;DH;SW;SH;PW;PH;RW;RH;_jit_;count;reuse count;prepare, ms;process, ms;code size, B\n" );
            for( auto& item : res ) {
                printf( "%d;%d;%d;%d;%d;%d;%d;%d;%d;%d;%d;%d;_jit_;%lu;%lu;%.4f;%.4f;%lu\n",
                        item.first.filterCount,
                        item.first.channelCount,
                        item.first.filterHeight,
                        item.first.filterWidth,
                        item.first.dilationHeight,
                        item.first.dilationWidth,
                        item.first.strideHeight,
                        item.first.strideWidth,
                        item.first.paddingHeight,
                        item.first.paddingWidth,
                        item.first.resultHeight,
                        item.first.resultWidth,
                        item.second.count,
                        item.second.reuseCount,
                        std::chrono::duration_cast<us>( item.second.prepareMs ).count() / 1000.,
                        std::chrono::duration_cast<us>( item.second.processMs ).count() / 1000.,
                        item.second.codeSize );
            }
        }
    }

private:
    typedef std::chrono::high_resolution_clock Time;
    typedef std::chrono::microseconds us;
    typedef std::chrono::duration<float> fsec;

    CKey key;
    std::chrono::duration<float> prepareMs;
    std::chrono::time_point<std::chrono::system_clock> prepareMsStart;
    std::chrono::duration<float> processMs;
    std::chrono::time_point<std::chrono::system_clock> processMsStart;
    size_t codeSize;
#endif
private:
    static std::unordered_map<CKey, CResult, CKey::Hasher> res;
};

std::unordered_map<CJitDebug::CKey, CJitDebug::CResult, CJitDebug::CKey::Hasher> CJitDebug::res;

class CJitDebugHolder {
public:
    CJitDebugHolder() = default;
    ~CJitDebugHolder() {
#ifdef JIT_DEBUG
        CJitDebug::PrintResult();
#endif
    }
};

static CJitDebugHolder jitDebugHolder;

} // namespace NeoML
