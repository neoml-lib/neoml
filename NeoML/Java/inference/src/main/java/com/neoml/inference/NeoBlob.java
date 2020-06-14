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

import java.nio.Buffer;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.IntBuffer;
import java.nio.FloatBuffer;

/**
 * A blob of data that can be passed into the network and received from the network output
 * <p>A blob stores the data in the memory of the processing device used by the math engine.
 * Call close after you are done with the blob to release the resources allocated in the native library.</p>
 */
public final class NeoBlob implements AutoCloseable {
    /** The type of data in a blob */
    public enum Type { FLOAT32, INT32 }

    /**
     * Creates a data blob
     *
     * @param mathEngine The math engine that will provide the memory to store the blob data
     * @param blobType The type of data in the blob
     * @param batchLength The sequence length (number of objects)
     * @param batchWidth The number of sequences processed together
     * @param height Object height
     * @param width Object width
     * @param depth Object depth
     * @param channelCount The number of channels
     *
     * @return The data blob
     */
    public static NeoBlob CreateDnnBlob(NeoMathEngine mathEngine, Type blobType, int batchLength, int batchWidth, int height, int width, int depth, int channelCount) {
        return new NeoBlob( mathEngine, blobType, batchLength, batchWidth, height, width, depth, channelCount );
    }

    /** The math engine for the blob */
    public NeoMathEngine GetMathEngine() { return mathEngine; }

    /** Data type of the blob */
    public Type GetType() { return type; }

    /** Sequence length */
    public int GetBatchLength() { return batchLength; }

    /** Batch size */
    public int GetBatchWidth() { return batchWidth; }

     /** Height */
    public int GetHeight() { return height; }

     /** Width */
    public int GetWidth() { return width; }

     /** Depth */
    public int GetDepth() { return depth; }

     /** The number of channels */
    public int GetChannelsCount() { return channelsCount; }

    /**
     * Writes the data into the blob
     * We recommend a direct buffer for best performance
     *
     * @param data The data to be written
     */
    public void SetData(Buffer data) {
        if( data instanceof ByteBuffer ) {
            if( !data.isDirect() ) {
                ByteBuffer buff = ByteBuffer.allocateDirect(data.capacity());
                buff.order(ByteOrder.nativeOrder());
                buff.put((ByteBuffer)data);
                data.rewind();
                buff.rewind();
                data = buff;
            }
        } else if( data instanceof FloatBuffer ) {
            if( GetType() != Type.FLOAT32 ) {
                throw new IllegalArgumentException("Can't set a float buffer to the non-float blob.");
            }
            if( !data.isDirect() ) {
                ByteBuffer buff = ByteBuffer.allocateDirect(data.capacity() * (Float.SIZE / Byte.SIZE));
                buff.order(ByteOrder.nativeOrder());
                buff.asFloatBuffer().put((FloatBuffer)data);
                data.rewind();
                buff.rewind();
                data = buff;
            }
        } else if( data instanceof IntBuffer ) {
            if( GetType() != Type.INT32 ) {
                throw new IllegalArgumentException("Can't set an int buffer to the non-int blob.");
            }
            if( !data.isDirect() ) {
                ByteBuffer buff = ByteBuffer.allocateDirect(data.capacity() * (Integer.SIZE / Byte.SIZE));
                buff.order(ByteOrder.nativeOrder());
                buff.asIntBuffer().put((IntBuffer)data);
                data.rewind();
                buff.rewind();
                data = buff;
            }
        } else {
            throw new IllegalArgumentException("Unexpected buffer type: " + data);
        }
        setData( nativeHandle, data );
    }

    /** Retrieves the data from the blob
     *
     * @return Returns a buffer with the data
     */
    public ByteBuffer GetData() {
        if( nativeHandle == 0 ) {
            throw new IllegalArgumentException("The blob has already been closed!");
        }
        ByteBuffer result = ByteBuffer.allocateDirect(getDataSize( nativeHandle ));
        result.order(ByteOrder.nativeOrder());
        getData( nativeHandle, result );
        return result;
    }

    /** Releases the resources allocated on the native library side */
    @Override
    public void close() {
        destroyDnnBlob( nativeHandle );
        nativeHandle = 0;
    }

    /** Reserved for internal use */
    public long GetNativeHandle() { return nativeHandle; }

    @Override
    protected void finalize() {
        close();
    }

    /** Reserved for internal use */
    public NeoBlob( NeoMathEngine mathEngine, long nativeHandle ) {
        this.nativeHandle = nativeHandle;
        this.mathEngine = mathEngine;
        this.type = Type.values()[getType( nativeHandle )];
        this.batchLength = getBatchLength( nativeHandle );
        this.batchWidth = getBatchWidth( nativeHandle );
        this.height = getHeight( nativeHandle );
        this.width = getWidth( nativeHandle );
        this.depth = getDepth( nativeHandle );
        this.channelsCount = getChannelsCount( nativeHandle );
    }

    private long nativeHandle = 0;
    private NeoMathEngine mathEngine;
    private Type type;
    private int batchLength;
    private int batchWidth;
    private int height;
    private int width;
    private int depth;
    private int channelsCount;

    private NeoBlob(NeoMathEngine mathEngine, Type type, int batchLength, int batchWidth, int height, int width, int depth, int channelCount) {
        if( mathEngine.GetNativeHandle() == 0 ) {
            throw new IllegalArgumentException("The mathEngine has already been closed!");
        }
        this.nativeHandle = createDnnBlob( mathEngine.GetNativeHandle(), type.ordinal(), batchLength, batchWidth, height, width, depth, channelCount );
        this.mathEngine = mathEngine;
        this.type = type;
        this.batchLength = batchLength;
        this.batchWidth = batchWidth;
        this.height = height;
        this.width = width;
        this.depth = depth;
        this.channelsCount = channelCount;
    }

    private native long createDnnBlob( long mathEngineNativeHandle, int type, int batchLength, int batchWidth, int height, int width, int depth, int channelCount );
    private native void destroyDnnBlob( long nativeHandle );
    private native int getType( long nativeHandle );
    private native int getBatchLength( long nativeHandle );
    private native int getBatchWidth( long nativeHandle );
    private native int getHeight( long nativeHandle );
    private native int getWidth( long nativeHandle );
    private native int getDepth( long nativeHandle );
    private native int getChannelsCount( long nativeHandle );
    private native int getDataSize( long nativeHandle );
    private native void setData( long nativeHandle, Buffer data );
    private native void getData( long nativeHandle, Buffer data );
}