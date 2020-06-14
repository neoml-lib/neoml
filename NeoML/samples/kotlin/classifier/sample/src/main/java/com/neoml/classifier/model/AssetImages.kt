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

import android.content.res.AssetManager
import com.neoml.classifier.App

fun getImagesFromAsset(): List<String> = mutableListOf<String>()
    .apply {
        val assetManager = App.appContext.assets
        addAll(assetManager.getFilesPath("doc", "jpg"))
        addAll(assetManager.getFilesPath("non_doc", "jpg"))
    }.toList()

private fun AssetManager.getFilesPath(directoryName: String, type: String): List<String> =
    this.list(directoryName)
        .orEmpty()
        .filter { it.contains(".$type") }
        .map { "$directoryName/$it" }
