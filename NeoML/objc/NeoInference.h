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

#import <Foundation/Foundation.h>

#ifdef __cplusplus
#define NEOML_API_LINKAGE extern "C" __attribute__((visibility("default")))
#else
#define NEOML_API_LINKAGE extern __attribute__((visibility("default")))
#endif

NS_ASSUME_NONNULL_BEGIN

// Error domain of NeoML.
FOUNDATION_EXPORT NSString *const NeoErrorDomain;

/// NeoML error codes.
typedef NS_ENUM(NSUInteger, NeoErrorCode) {
     /// No error
    NeoErrorOK = 0,
    /// Internal error, details in error description
    NeoErrorCodeInternalError,
    /// No GPU available
    NeoErrorCodeNoAvailableGPU,
    /// No CPU available
    NeoErrorCodeNoAvailableCPU,
    /// Parameter value is invalid
    NeoErrorCodeInvalidParameter,
    /// Error when running the network, details in error description
    NeoErrorCodeRunDnnError,
    /// Error when loading the network, details in error description
    NeoErrorCodeLoadDnnError
};

//----------------------------------------------------------------------------------------------------------

/// The type of processing unit used for the math engine
typedef NS_ENUM(NSUInteger, NeoMathEngineType) {
    // CPU
    NeoMathEngineTypeCpu = 0,
    // GPU
    NeoMathEngineTypeGpu = 1
};

//----------------------------------------------------------------------------------------------------------

/// The math engine
NEOML_API_LINKAGE
@interface NeoMathEngine : NSObject

/// Unavailable
- (instancetype)init NS_UNAVAILABLE;

/// Creates a math engine that uses a CPU for calculations
///
/// @param threadCount The number of threads that may be used (set to 0 to automatically start as many threads as the CPU has cores)
/// @param error An optional parameter; will only be filled in if an error occurs
///
/// @return The math engine or nil in case of error
+ (nullable instancetype)createCPUMathEngine:(NSUInteger)threadCount
                                       error:(NSError**)error;

/// Creates a math engine that uses a GPU for calculations
///
/// @param error An optional parameter; will only be filled in if an error occurs
///
/// @return The math engine or nil in case of error
+ (nullable instancetype)createGPUMathEngine:(NSError**)error;

/// Math engine type
@property(nonatomic, readonly) NeoMathEngineType mathEngineType;

@end // MathEngine

//----------------------------------------------------------------------------------------------------------

/// Type of data in a blob
typedef NS_ENUM(NSUInteger, NeoBlobType) {
    /// float32
    NeoBlobTypeFloat32 = 1,
    /// int32
    NeoBlobTypeInt32 = 2
};

//----------------------------------------------------------------------------------------------------------

/// A data blob that is passed into the network and received from the network output
/// A blob stores the data in the memory of the processing device used by the math engine
NEOML_API_LINKAGE
@interface NeoBlob : NSObject

/// Unavailable
- (instancetype)init NS_UNAVAILABLE;

/// Creates a data blob
///
/// @param mathEngine The math engine that will provide the memory to store the blob data
/// @param blobType The type of data in the blob
/// @param batchLength The sequence length (number of objects)
/// @param batchWidth The number of sequences processed together
/// @param height Object height
/// @param width Object width
/// @param depth Object depth
/// @param channelCount The number of channels
/// @param error An optional parameter; will only be filled in if an error occurs
///
/// @return The data blob or nil in case of error
+ (nullable instancetype)createDnnBlob:(NeoMathEngine*)mathEngine
                              blobType:(NeoBlobType)blobType
	                       batchLength:(NSUInteger)batchLength
                            batchWidth:(NSUInteger)batchWidth
                                height:(NSUInteger)height
                                 width:(NSUInteger)width
                                 depth:(NSUInteger)depth
                          channelCount:(NSUInteger)channelCount 
                                 error:(NSError**)error;

/// The math engine for the blob
@property(nonatomic, readonly) NeoMathEngine* mathEngine;

/// Data type of the blob
@property(nonatomic, readonly) NeoBlobType blobType;

/// Sequence length
@property(nonatomic, readonly) NSUInteger batchLength;

/// Batch size
@property(nonatomic, readonly) NSUInteger batchWidth;

/// Height
@property(nonatomic, readonly) NSUInteger height;

/// Width
@property(nonatomic, readonly) NSUInteger width;

/// Depth
@property(nonatomic, readonly) NSUInteger depth;

/// The number of channels
@property(nonatomic, readonly) NSUInteger channelCount;

/// Retrieves the data from the blob
///
/// @param error An optional parameter; will only be filled in if an error occurs
///
/// @return The data or nil in case of error
- (nullable NSData*)getData:(NSError**)error;

/// Writes the data into the blob
///
/// @param data The data to be written
/// @param error An optional parameter; will only be filled in if an error occurs
///
/// @return Whether the data was written successfully
- (BOOL)setData:(NSData*)data
          error:(NSError**)error;

@end // NeoBlob

//----------------------------------------------------------------------------------------------------------

/// Neural network
NEOML_API_LINKAGE
@interface NeoDnn : NSObject

/// Unavailable
- (instancetype)init NS_UNAVAILABLE;

/// Loads the network from the buffer
///
/// @param mathEngine The math engine that will be used for all operations with the network
/// @param data The memory buffer that contains the network description
/// @param error An optional parameter; will only be filled in if an error occurs
///
/// @return The loaded network or nil in case of error
+ (nullable instancetype)createDnn:(NeoMathEngine*)mathEngine
                              data:(NSData*)data
                             error:(NSError**)error;

/// Loads the network from the buffer
///
/// @param mathEngine The math engine that will be used for all operations with the network
/// @param data The memory buffer that contains the network description in ONNX format
/// @param error An optional parameter; will only be filled in if an error occurs
///
/// @return The loaded network or nil in case of error
+ (nullable instancetype)createDnnFromOnnx:(NeoMathEngine*)mathEngine
                                      data:(NSData*)data
                                     error:(NSError**)error;

/// The math engine for the network
@property(nonatomic, readonly) NeoMathEngine* mathEngine;

/// The number of network inputs
@property(nonatomic, readonly) NSUInteger inputCount;

/// The number of network outputs
@property(nonatomic, readonly) NSUInteger outputCount;

/// Gets the name of the input by its index
///
/// @param index The index of the input, in the [0, inputCount) range
/// @param error An optional parameter; will only be filled in if an error occurs
///
/// @return The name of the input or nil in case of error
- (nullable NSString*)getInputName:(NSUInteger)index
                             error:(NSError**)error;

/// Sets the data blob that will be passed into the given input
///
/// @param index The index of the input, in the [0, inputCount) range
/// @param error An optional parameter; will only be filled in if an error occurs
///
/// @return Whether the input data was set successfully
- (BOOL)setInputBlob:(NSUInteger)index
                blob:(NeoBlob*)blob
               error:(NSError**)error;

/// Runs the network with the previously set inputs
///
/// @param error An optional parameter; will only be filled in if an error occurs
///
/// @return Whether the operation was successful
- (BOOL)run:(NSError**)error;

/// Gets the name of the output by its index
///
/// @param index The index of the output, in the [0, outputCount) range
/// @param error An optional parameter; will only be filled in if an error occurs
///
/// @return The name of the output or nil in case of error
- (nullable NSString*)getOutputName:(NSUInteger)index
                              error:(NSError**)error;

/// Retrieves the data blob from the specified output
/// The data will be valid until the next run
///
/// @param index The index of the output, in the [0, outputCount) range
/// @param error An optional parameter; will only be filled in if an error occurs
///
/// @return The output data blob or nil in case of error
- (nullable NeoBlob*)getOutputBlob:(NSUInteger)index
                              error:(NSError**)error;

@end // NeoDnn

NS_ASSUME_NONNULL_END
