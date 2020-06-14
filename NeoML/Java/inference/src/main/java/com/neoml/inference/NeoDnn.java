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

package com.neoml.inference;

import java.io.IOException;
import java.io.InputStream;
import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * Neural network
 * <p>Call close after use to release the resources allocated in the native library</p>
 */
public final class NeoDnn implements AutoCloseable {
    /** Loads the network from the buffer
     * If the buffer is direct, the data will be copied from the buffer itself
     * Otherwise an intermediate buffer will be used to interact with the native library
     * @param mathEngine The math engine that will be used for all operations with the network
     * @param directBuffer The memory buffer that contains the network description
     *
     * @return The loaded network
     */
    public static NeoDnn CreateDnn(NeoMathEngine mathEngine, ByteBuffer directBuffer) {
        return new NeoDnn(mathEngine, directBuffer, false);
    }
    
    /** Loads the network in ONNX format from the buffer
     * If the buffer is direct, the data will be copied from the buffer itself
     * Otherwise an intermediate buffer will be used to interact with the native library
     * @param mathEngine The math engine that will be used for all operations with the network
     * @param directBuffer The memory buffer that contains the network description in ONNX format
     *
     * @return The loaded network
     */
    public static NeoDnn CreateDnnFromOnnx(NeoMathEngine mathEngine, ByteBuffer directBuffer) {
        return new NeoDnn(mathEngine, directBuffer, true);
    }

    /** The math engine for the network */
    public NeoMathEngine GetMathEngine() { return mathEngine; }

    /** The number of network inputs */
    public int GetInputsCount() { return inputsCount; }

    /** The number of network outputs */
    public int GetOutputsCount() { return outputsCount; }

    /** Gets the name of the input by its index
     *
     * @param index The index of the input, in the [0, inputCount) range
     *
     * @return The name of the input
     */
    public String GetInputName(int index) {
        if( index < 0 || index > inputsCount ) {
            throw new IllegalArgumentException("Invalid index.");
        }
        if( nativeHandle == 0 ) {
            throw new RuntimeException("The dnn has already been closed!");
        }
        return getInputName( nativeHandle, index);
    }

    /** Sets the data blob that will be passed into the given input
     *
     * @param index The index of the input, in the [0, inputCount) range
     */
    public void SetInputBlob(int index, NeoBlob blob) {
        if( index < 0 || index > inputsCount ) {
            throw new IllegalArgumentException("Invalid index.");
        }
        if( nativeHandle == 0 ) {
            throw new RuntimeException("The dnn has already been closed!");
        }
        setInputBlob( nativeHandle, index, blob.GetNativeHandle() );
    }

    /** Runs the network with the previously set inputs */
    public void Run() {
        if( nativeHandle == 0 ) {
            throw new RuntimeException("The dnn has already been closed!");
        }
        run( nativeHandle );
    }

    /** Gets the name of the output by its index
     *
     * @param index The index of the output, in the [0, outputCount) range
     *
     * @return The name of the output
     */
    public String GetOutputName(int index) {
        if( nativeHandle == 0 ) {
            throw new RuntimeException("The dnn has already been closed!");
        }
        return getOutputName( nativeHandle, index );
    }

    /** Retrieves the data blob from the specified output
     * The data will be valid until the next run
     *
     * @param index The index of the output, in the [0, outputCount) range
     *
     * @return The output data blob
     */
    public NeoBlob GetOutputBlob(int index) {
        if( nativeHandle == 0 ) {
            throw new RuntimeException("The dnn has already been closed!");
        }

        long blobNativeHandle = getOutputBlob( nativeHandle, index );
        return new NeoBlob( mathEngine, blobNativeHandle );
    }

    /** Releases the resources allocated on the native library side */
    @Override
    public void close() {
        destroyDnn( nativeHandle );
        nativeHandle = 0;
    }

    @Override
    protected void finalize() {
        close();
    }

    private long nativeHandle = 0;
    private NeoMathEngine mathEngine;
    private int inputsCount;
    private int outputsCount;

    private NeoDnn(NeoMathEngine dnnMathEngine, ByteBuffer directBuffer, boolean isOnnx) {
        if( directBuffer == null || !directBuffer.isDirect() || directBuffer.order() != ByteOrder.nativeOrder() ) {
            throw new IllegalArgumentException( "ByteBuffer should be a direct and has native byte order!" );
        }

        if( isOnnx ) {
            nativeHandle = createDnnFromOnnx( dnnMathEngine.GetNativeHandle(), directBuffer );
        } else {
            nativeHandle = createDnn( dnnMathEngine.GetNativeHandle(), directBuffer );
        }
        mathEngine = dnnMathEngine;
        inputsCount = getInputCount( nativeHandle );
        outputsCount = getOutputCount( nativeHandle );
    }

    private native long createDnn( long nativeHandle, Buffer directBuffer );
    private native long createDnnFromOnnx( long nativeHandle, Buffer directBuffer );
    private native void destroyDnn( long nativeHandle );
    private native String getInputName( long nativeHandle, int index );
    private native int getInputCount( long nativeHandle );
    private native void setInputBlob( long nativeHandle, int index, long blobNativeHandle );
    private native void run( long nativeHandle );
    private native int getOutputCount( long nativeHandle );
    private native String getOutputName( long nativeHandle, int index );
    private native long getOutputBlob( long nativeHandle, int index );
}