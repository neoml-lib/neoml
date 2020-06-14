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
 * <p>The math engine</p>
 * <p>Call close after use to release the resources allocated in the native library</p>
*/
public final class NeoMathEngine implements AutoCloseable {
    /** The type of processing unit used for the math engine */
    public enum Type { CPU, GPU }

    /**
     * Creates a math engine that uses a CPU for calculations
     *
     * @param threadCount The number of threads that may be used (set to 0 to automatically start as many threads as the CPU has cores)
     * @return The math engine
     */
    public static NeoMathEngine CreateCpuMathEngine(int threadCount) {
        return new NeoMathEngine( threadCount );
    }

    /** Creates a math engine that uses a GPU for calculations
     *
     * @return The math engine
     */
    public static NeoMathEngine CreateGpuMathEngine() {
        return new NeoMathEngine();
    }

    /** Gets the device type */
    public Type GetType() { return type; }

    /** Releases the resources allocated on the native library side */
    @Override
    public void close() {
        destroyMathEngine( nativeHandle );
        nativeHandle = 0;
    }

    /** Reserved for internal use */
    public long GetNativeHandle() { return nativeHandle; }

    @Override
    protected void finalize() {
        close();
    }

    private long nativeHandle = 0;
    private Type type;

    private NeoMathEngine() {
        this.nativeHandle = createGpuMathEngine();
        this.type = Type.GPU;
    }
    private NeoMathEngine( int threadCount ) {
        this.nativeHandle = createCpuMathEngine( threadCount );
        this.type = Type.CPU;
    }

    private native long createCpuMathEngine( int threadCount );
    private native long createGpuMathEngine();
    private native void destroyMathEngine( long nativeHandle );

    static {
        System.loadLibrary("NeoInferenceJni");
    }
}