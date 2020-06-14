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

import android.os.Handler
import com.neoml.classifier.App.Companion.mlEngine
import java.util.*
import java.util.concurrent.Executors

class MainModel : Observable() {

    private val threadPool = Executors.newFixedThreadPool(2)

    private val imagesFromAsset = getImagesFromAsset()

    companion object {
        private const val ASSET_PREDICT = "file:///android_asset/"
    }

    private val imageData: MutableList<ImageData> by lazy {
        mutableListOf<ImageData>().apply {
            for (imagePath in imagesFromAsset) {
                add(ImageData(imagePath = ASSET_PREDICT.plus(imagePath)))
                shuffle()
            }
        }
    }

    fun requestNonHandledData() {
        setChanged()
        notifyObservers(imageData.toList())
    }

    fun addImageFromGallery(imagePath: String?) {
        if (!imagePath.isNullOrEmpty()) {
            imageData.add(ImageData(imagePath = imagePath))
        }
    }

    fun classifyAllImages() {
        classify()
    }

    fun classify(imagePath: String, index: Int = 0, isSingleRequest: Boolean = false) {
        val handler = Handler()
        startClassify {
            synchronized(this) {
                val list = mutableListOf<ImageData>()
                imageData.forEach {
                    val data = if (it.imagePath == imagePath) {
                        ImageData(imagePath = it.imagePath, type = mlEngine.sendFileToClassify(it.imagePath))
                    } else it
                    list.add(data)
                }

                imageData.clear()
                imageData.addAll(list)

                handler.post {
                    setChanged()
                    notifyObservers(list)
                    if (!isSingleRequest) {
                        classify(pos = index + 1)
                    }
                }
            }
        }
    }

    private fun startClassify(runnable: () -> Unit) {
        threadPool.submit(runnable)
    }

    private fun classify(pos: Int = 0) {
        if (pos == imageData.count()) return
        classify(imagePath = imageData[pos].imagePath, index = pos)
    }

    fun destroy() {
        threadPool.shutdownNow()
    }
}