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

/// Classification Engine Delegate
/// Send item update information
protocol ClassificationEngineDelegate {
    func classificationEngine(_ engine: ClassificationEngine, didUpdate item: DocumentItem)
}

/// Handle image recognition and conversion
 class ClassificationEngine {
        
    private (set) var classifier: ImageClassifier!
    private (set) var converter: ImageConverter!
    var delegate: ClassificationEngineDelegate?
    
    init(classifier: ImageClassifier, converter: ImageConverter) {
        self.classifier = classifier
        self.converter = converter
    }
    
    /// Recognize selected item documetType with completion block
    /// - Parameters:
    ///   - item: DocumentItem
    ///   - completion: recognition completion block
    func classify(_ item: DocumentItem, completion:@escaping () -> ()) {
        converter.rawData(for: item.image) { [unowned self] result in
            guard let data = result else {
                item.documentType = .unrecognized
                completion()
                return
            }
            self.classifier.classify(rawImage: data) { result in
                item.documentType = result
                self.notify(item)
                completion()
            }
        }
    }
        
    /// Recognize DocumentItem items array
    /// - Parameters:
    ///   - items: DocumentItems array
    func classify(_ items: [DocumentItem]) {
        DispatchQueue.global(qos: .utility).async { [weak self] in
            for item in items {
                self?.classify(item) {
                    self?.notify(item)}
            }
        }
    }
    
    /// Notifies delegate about document Item recognition
    /// - Parameter item: documentItem
    func notify(_ item: DocumentItem) {
        DispatchQueue.main.async {
            self.delegate?.classificationEngine(self, didUpdate: item)
        }
    }
    
}
