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

package com.neoml.classifier.main

import com.neoml.classifier.model.ImageData
import java.util.*

/**
 * This listener must provide a binding between [MainActivity] and its child fragments.
 */
interface ActivityListener {
    /** Allows show/hide camera and gallery buttons  */
    fun changeBottomButtons(isDetail: Boolean)
    /** Override RUN button action */
    fun bindRunAction(imageData: ImageData? = null)
    /** Request images without classification */
    fun requestData()

    /** Add an observer to receive data after classify */
    fun addObserver(observer: Observer)
    fun deleteObserver(observer: Observer)

}