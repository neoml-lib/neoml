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

#include <NeoInference.h>
#include <NeoProxy/NeoProxy.h>

@import Foundation;

NS_ASSUME_NONNULL_BEGIN

//------------------------------------------------------------------------------------------------------------

// Error domain of NeoML.
FOUNDATION_EXPORT NSString *const NeoErrorDomain = @"org.neoml";

static inline void createErrorWithCode( NeoErrorCode code, NSString* text, NSError** error )
{
    if( error ) {
        *error = [NSError errorWithDomain:NeoErrorDomain
                                     code:code
                                 userInfo:@{NSLocalizedDescriptionKey : text}];
    }
}

static inline void createErrorWithInfo( const CDnnErrorInfo& errorInfo, NSError** error )
{
    if( error ) {
        NSString* description = [NSString stringWithUTF8String:(const char*)errorInfo.Description];
        *error = [NSError errorWithDomain:NeoErrorDomain
                                     code:(NeoErrorCode)errorInfo.Type
                                 userInfo:@{NSLocalizedDescriptionKey : description}];
    }
}

//------------------------------------------------------------------------------------------------------------

@implementation NeoMathEngine {
    const CDnnMathEngineDesc* desc;
}

+ (nullable instancetype)createCPUMathEngine:(NSUInteger)threadCount
                                       error:(NSError**)error
{
    CDnnErrorInfo errorInfo;
    const CDnnMathEngineDesc* mathEngineDesc = CreateCPUMathEngine( (int)threadCount, &errorInfo );
    if( mathEngineDesc == 0 ) {
        createErrorWithInfo( errorInfo, error );
        return nil;
    }
    NeoMathEngine* mathEngine = [[NeoMathEngine alloc] initWithDesc:mathEngineDesc];
    if( mathEngine == nil ) {
        createErrorWithCode( NeoErrorCodeInternalError, @"Failed to allocate a NeoMathEngine object.", error );
        DestroyMathEngine( mathEngineDesc );
        return nil;
    }
    return mathEngine;
}

+ (nullable instancetype)createGPUMathEngine:(NSError**)error
{
    CDnnErrorInfo errorInfo;
    const CDnnMathEngineDesc* mathEngineDesc = CreateGPUMathEngine( &errorInfo );
    if( mathEngineDesc == 0 ) {
        createErrorWithInfo( errorInfo, error );
        return nil;
    }
    NeoMathEngine* mathEngine = [[NeoMathEngine alloc] initWithDesc:mathEngineDesc];
    if( mathEngine == nil ) {
        createErrorWithCode( NeoErrorCodeInternalError, @"Failed to allocate a NeoMathEngine object.", error );
        DestroyMathEngine( mathEngineDesc );
        return nil;
    }
    return mathEngine;
}

- (instancetype)initWithDesc:(const CDnnMathEngineDesc*)mathEngineDesc
{
    self = [super init];
    if(self != nil) {
        desc = mathEngineDesc;
        _mathEngineType = (NeoMathEngineType)desc->Type;
    }
    return self;
}

- (void)dealloc
{
    if( desc != 0 ) {
        DestroyMathEngine( desc );
        desc = 0;
    }
}

- (const CDnnMathEngineDesc*)rawPtr
{
    return desc;
}

@end

//----------------------------------------------------------------------------------------------------------

@implementation NeoBlob {
    const CDnnBlobDesc* desc;
}

+ (nullable instancetype)createDnnBlob:(NeoMathEngine*)mathEngine
                              blobType:(NeoBlobType)blobType
                           batchLength:(NSUInteger)batchLength
                            batchWidth:(NSUInteger)batchWidth
                                height:(NSUInteger)height
                                 width:(NSUInteger)width
                                 depth:(NSUInteger)depth
                          channelCount:(NSUInteger)channelCount 
                                 error:(NSError**)error
{
    if( mathEngine == nil ) {
        createErrorWithCode( NeoErrorCodeInvalidParameter, @"Invalid parameter mathEngine.", error ); 
        return nil;
    }
    CDnnErrorInfo errorInfo;
    const CDnnBlobDesc* dnnBlobDesc = CreateDnnBlob( [mathEngine rawPtr], (TDnnBlobType)blobType,
        (int)batchLength, (int)batchWidth, (int)height, (int)width, (int)depth, (int)channelCount, &errorInfo );
    if( dnnBlobDesc == 0 ) {
        createErrorWithInfo( errorInfo, error );
        return nil;
    }
    NeoBlob* blob = [[NeoBlob alloc] initWithDesc:dnnBlobDesc mathEngine:mathEngine];
    if( blob == nil ) {
        createErrorWithCode( NeoErrorCodeInternalError, @"Failed to allocate a NeoBlob object.", error );
        DestroyDnnBlob( dnnBlobDesc );
        return nil;
    }
    return blob;
}

- (instancetype)initWithDesc:(const CDnnBlobDesc*)blobDesc mathEngine:(NeoMathEngine*)mathEngine
{
    self = [super init];
    if(self != nil) {
        desc = blobDesc;
        _mathEngine = mathEngine;
        _blobType = (NeoBlobType)blobDesc->Type;
        _batchLength = blobDesc->BatchLength;
        _batchWidth = blobDesc->BatchWidth;
        _height = blobDesc->Height;
        _width = blobDesc->Width;
        _depth = blobDesc->Depth;
        _channelCount = blobDesc->ChannelCount;
    }
    return self;
}

- (void)dealloc
{
    if( desc != 0 ) {
        DestroyDnnBlob( desc );
        desc = 0;
    }
}

- (const CDnnBlobDesc*)rawPtr
{
    return desc;
}

- (nullable NSData*)getData:(NSError**)error
{
    NSMutableData* result = [[NSMutableData alloc] initWithCapacity:desc->DataSize];
    if( result == nil ) {
        createErrorWithCode( NeoErrorCodeInternalError, @"Failed to allocate the NSData. Not enough memory.", error );
        return nil;
    }
    result.length = desc->DataSize;

    CDnnErrorInfo errorInfo;
    if( !CopyFromBlob( [result mutableBytes], desc, &errorInfo ) ) {
        createErrorWithInfo( errorInfo, error );
        return nil;
    }
    return result;
}

- (BOOL)setData:(NSData*)data
          error:(NSError**)error
{
    if( data == nil ) {
        createErrorWithCode( NeoErrorCodeInvalidParameter, @"Invalid parameter data.", error ); 
        return false;
    }
    if((int)[data length] != desc->DataSize ) {
        createErrorWithCode( NeoErrorCodeInvalidParameter, @"Wrong data size.", error ); 
        return false;
    }

    CDnnErrorInfo errorInfo;
    if( !CopyToBlob( desc, [data bytes], &errorInfo ) ) {
        createErrorWithInfo( errorInfo, error );
        return false;
    }
    return true;
}

@end

//----------------------------------------------------------------------------------------------------------

@implementation NeoDnn {
    const CDnnDesc* desc;
}

+ (nullable instancetype)createDnn:(NeoMathEngine*)mathEngine
                              data:(NSData*)data
                             error:(NSError**)error
{
    if( mathEngine == nil ) {
        createErrorWithCode( NeoErrorCodeInvalidParameter, @"Invalid parameter mathEngine.", error ); 
        return nil;
    }
    if( data == nil || [data length] > 1024 * 1024 * 1024 ) {
        createErrorWithCode( NeoErrorCodeInvalidParameter, @"Invalid parameter data.", error ); 
        return nil;
    }

    CDnnErrorInfo errorInfo;
    const CDnnDesc* dnnDesc = CreateDnnFromBuffer( [mathEngine rawPtr], [data bytes], (int)[data length], &errorInfo );
    if( dnnDesc == 0 ) {
        createErrorWithInfo( errorInfo, error );
        return nil;
    }
    NeoDnn* dnn = [[NeoDnn alloc] initWithDesc:dnnDesc mathEngine: mathEngine];
    if( dnn == nil ) {
        createErrorWithCode( NeoErrorCodeInternalError, @"Failed to create the dnn. Not enough memory.", error );
        DestroyDnn( dnnDesc );
        return nil;
    }
    return dnn;
}

+ (nullable instancetype)createDnnFromOnnx:(NeoMathEngine*)mathEngine
                                      data:(NSData*)data
                                     error:(NSError**)error
{
    if( mathEngine == nil ) {
        createErrorWithCode( NeoErrorCodeInvalidParameter, @"Invalid parameter mathEngine.", error ); 
        return nil;
    }
    if( data == nil || [data length] > 1024 * 1024 * 1024 ) {
        createErrorWithCode( NeoErrorCodeInvalidParameter, @"Invalid parameter data.", error ); 
        return nil;
    }

    CDnnErrorInfo errorInfo;
    const CDnnDesc* dnnDesc = CreateDnnFromOnnxBuffer( [mathEngine rawPtr], [data bytes], (int)[data length], &errorInfo );
    if( dnnDesc == 0 ) {
        createErrorWithInfo( errorInfo, error );
        return nil;
    }
    NeoDnn* dnn = [[NeoDnn alloc] initWithDesc:dnnDesc mathEngine: mathEngine];
    if( dnn == nil ) {
        createErrorWithCode( NeoErrorCodeInternalError, @"Failed to create the dnn. Not enough memory.", error );
        DestroyDnn( dnnDesc );
        return nil;
    }
    return dnn;
}

- (instancetype)initWithDesc:(const CDnnDesc*)dnnDesc mathEngine:(NeoMathEngine*)mathEngine
{
    self = [super init];
    if( self != nil ) {
        _inputCount = dnnDesc->InputCount;
        _outputCount = dnnDesc->OutputCount;
        _mathEngine = mathEngine;
        desc = dnnDesc;
    }
    return self;
}

- (void)dealloc
{
    if( desc != 0 ) {
        DestroyDnn( desc );
        desc = 0;
    }
}

- (nullable NSString*)getInputName:(NSUInteger)index
                             error:(NSError**)error
{
    CDnnErrorInfo errorInfo;
    const char* name = GetInputName( desc, (int)index, &errorInfo );
    if( name == 0 ) {
        createErrorWithInfo( errorInfo, error );
        return nil;
    }
    NSString* result = [NSString stringWithUTF8String:name];
    if( result == nil ) {
        createErrorWithCode( NeoErrorCodeInternalError, @"Failed to create the string. Not enough memory.", error );
        return nil;
    }
    return result;
}

- (BOOL)setInputBlob:(NSUInteger)index
                blob:(NeoBlob*)blob
               error:(NSError**)error
{
    if( blob == nil ) {
        createErrorWithCode( NeoErrorCodeInvalidParameter, @"Invalid parameter blob.", error ); 
        return nil;
    }

    const CDnnBlobDesc* blobDesc = [blob rawPtr];
    if( blobDesc == nil ) {
        createErrorWithCode( NeoErrorCodeInvalidParameter, @"Invalid parameter blob.", error );
        return false;
    }

    CDnnErrorInfo errorInfo;
    if( !SetInputBlob( desc, (int)index, blobDesc, &errorInfo ) ) {
        createErrorWithInfo( errorInfo, error );
        return false;
    }
    return true;
}

- (BOOL)run:(NSError**)error
{
    CDnnErrorInfo errorInfo;
    if( !DnnRunOnce( desc, &errorInfo ) ) {
        createErrorWithInfo( errorInfo, error );
        return false;
    }
    return true;
}

- (nullable NSString*)getOutputName:(NSUInteger)index
                              error:(NSError**)error
{
    CDnnErrorInfo errorInfo;
    const char* name = GetOutputName( desc, (int)index, &errorInfo );
    if( name == 0 ) {
        createErrorWithInfo( errorInfo, error );
        return nil;
    }
    return [NSString stringWithUTF8String:name];
}

- (nullable NeoBlob*)getOutputBlob:(NSUInteger)index
                              error:(NSError**)error
{
    CDnnErrorInfo errorInfo;
    const CDnnBlobDesc* dnnBlobDesc = GetOutputBlob( desc, (int)index, &errorInfo );
    if( dnnBlobDesc == 0 ) {
        createErrorWithInfo( errorInfo, error );
        return nil;
    }
    NeoBlob* blob = [[NeoBlob alloc] initWithDesc:dnnBlobDesc mathEngine:_mathEngine];
    if( blob == nil ) {
        createErrorWithCode( NeoErrorCodeInternalError, @"Failed to allocate a NeoBlob object.", error );
        DestroyDnnBlob( dnnBlobDesc );
        return nil;
    }
    return blob;
}

@end

NS_ASSUME_NONNULL_END
