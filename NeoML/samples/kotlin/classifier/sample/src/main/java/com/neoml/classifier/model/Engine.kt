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

package com.neoml.classifier.model

import android.graphics.Bitmap
import android.net.Uri
import com.bumptech.glide.Glide
import com.bumptech.glide.load.DecodeFormat
import com.neoml.inference.NeoBlob
import com.neoml.inference.NeoBlob.CreateDnnBlob
import com.neoml.inference.NeoDnn.CreateDnn
import com.neoml.inference.NeoMathEngine
import com.neoml.classifier.App
import java.io.ByteArrayOutputStream
import java.io.IOException
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer


class Engine {

    /** Model filename */
    private val modelName = "TextNotTextClassifier.dnn"

    private val channelsCount = 3

    private val widthImage = 224
    private val heightImage = 224

    private val neoMathEngine by lazy {
        NeoMathEngine.CreateCpuMathEngine(1)
    }

    private val neoDnn by lazy {
        val model = loadModelFile()
        CreateDnn(neoMathEngine, model)
    }

    @Throws(IOException::class)
    private fun loadModelFile(): ByteBuffer {
        return App.appContext.assets.open(modelName).use { input ->
            val available = input.available()
            val output = ByteArrayOutputStream(available)
            val tmp = ByteArray(1024)

            var r = input.read(tmp)
            while (r != -1) {
                output.write(tmp, 0, r)
                r = input.read(tmp)
            }
            val byteBuffer = ByteBuffer.allocateDirect(available)
            byteBuffer.order(ByteOrder.nativeOrder())
            byteBuffer.put(output.toByteArray())
            byteBuffer
        }
    }

    companion object {
        fun getInstance(): Engine = HOLDER.instance
    }

    private object HOLDER {
        val instance = Engine()
    }

    /**
     * Send file to handle by fml2
     * @param filePath should to use for obtaining necessary data format
     * @return coefficient kind 0,1,2,..10
     */
    fun sendFileToClassify(filePath: String): Type {
        val bitmap = Glide.with(App.appContext)
            .asBitmap()
            .load(Uri.parse(filePath))
            .fitCenter()
            .format(DecodeFormat.PREFER_ARGB_8888)
            .override(widthImage, heightImage)
            .submit()
            .get()

        val width = bitmap.width
        val height = bitmap.height

        val blob: NeoBlob = CreateDnnBlob(
            neoMathEngine,
            NeoBlob.Type.FLOAT32,
            1, 1,
            height, width,
            1, channelsCount
        )

        val buff = convertBitmapToByteBuffer(
            bitmap = bitmap,
            width = width,
            height = height
        )
        blob.SetData(buff)
        neoDnn.SetInputBlob(0, blob)
        /** start NeoMathEngine */
        neoDnn.Run()

        val outputBlob = neoDnn.GetOutputBlob(0)
        val result = outputBlob.GetData().asFloatBuffer()
        val argMax = getArgMax(result)

        return Type.values()[argMax]
    }

    private fun convertBitmapToByteBuffer(
        bitmap: Bitmap,
        width: Int,
        height: Int,
        pixelSize: Int = 4,
        channels: Int = channelsCount
    ): ByteBuffer {
        /* Create ByteBuffer for predict */
        val pixelCount = height * width
        val byteBuffer = ByteBuffer.allocateDirect(
            1 * pixelCount * pixelSize * channels
        )
        byteBuffer.rewind()
        byteBuffer.order(ByteOrder.nativeOrder())
        val intValues = IntArray(pixelCount)
        bitmap.getPixels(intValues, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)

        // Convert the image to floating point.
        for (pixel in 0 until pixelCount) {
            val value = intValues[pixel]
            byteBuffer.putFloat((value shr 16 and 0xFF).toFloat() / 255f)
            byteBuffer.putFloat((value shr 8 and 0xFF).toFloat() / 255f)
            byteBuffer.putFloat((value and 0xFF).toFloat() / 255f)
        }

        return byteBuffer
    }

    /**
     * Returns the indices of the maximum values along an float array
     * @param input FloatBuffer
     */
    private fun getArgMax(input: FloatBuffer): Int {
        val array = FloatArray(input.capacity())
        for (index in 0 until input.capacity()) {
            array[index] = input.get(index)
        }
        var res = 0
        var max = array[0]

        val n = array.size
        for (i in 0 until n) {
            val value = array[i]
            if (value > max) {
                res = i
                max = value
            }
        }
        return res
    }
}
