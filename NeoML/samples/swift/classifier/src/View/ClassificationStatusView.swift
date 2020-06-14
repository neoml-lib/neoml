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

import UIKit

/// Show image recognition status
class ClassificationStatusView: UIView {
    
    @IBOutlet private weak var indicator: UIView!
    @IBOutlet private weak var classLabel: UILabel!

    /// Show image document type
    /// - Parameter documentType: image recognized type
    func showStatus(status documentType: DocumentType) {
        indicator.backgroundColor =  documentType.color
        classLabel.textColor = documentType.textColor
        classLabel.text = documentType.description
    }
}

