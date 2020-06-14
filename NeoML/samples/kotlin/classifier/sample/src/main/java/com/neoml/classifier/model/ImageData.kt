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

import android.os.Parcel
import android.os.Parcelable

data class ImageData(
    val imagePath: String,
    var type: Type = Type.UNCLASSIFIED
) : Parcelable {

    constructor(parcel: Parcel) : this(
        imagePath = parcel.readString() as String,
        type = parcel.readSerializable() as Type
    )

    override fun writeToParcel(parcel: Parcel, flags: Int) {
        parcel.writeString(imagePath)
        parcel.writeSerializable(type)
    }

    override fun describeContents(): Int {
        return 0
    }

    companion object CREATOR : Parcelable.Creator<ImageData> {
        override fun createFromParcel(parcel: Parcel): ImageData {
            return ImageData(parcel)
        }

        override fun newArray(size: Int): Array<ImageData?> {
            return arrayOfNulls(size)
        }
    }
}