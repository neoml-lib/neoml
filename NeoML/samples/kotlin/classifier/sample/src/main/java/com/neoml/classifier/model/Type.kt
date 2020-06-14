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

import androidx.annotation.DrawableRes
import androidx.annotation.StringRes
import com.neoml.classifier.R

enum class Type(@StringRes val textId: Int, @DrawableRes val drawableId: Int) {
    TEXT(R.string.text_type, R.drawable.text_bubble),
    NON_DOCUMENT(R.string.non_document_type, R.drawable.non_text_bubble),
    UNCLASSIFIED(R.string.unclassified_type, R.drawable.unclassified_bubble)
}
