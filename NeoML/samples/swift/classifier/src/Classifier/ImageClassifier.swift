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

import Foundation

/// Classify images with NeoML
class ImageClassifier {
    
    /// TextNotTextClassifier neural network has image input with 224x224 size
    /// Output is array [float 1, float 2] where array argmax is the text/non text output
    let networkFileData: Data = {
        let cnnFilePath = Bundle.main.path(forResource: "TextNotTextClassifier", ofType: "dnn", inDirectory: "data")!
        let cnnFileData = FileManager.default.contents(atPath: cnnFilePath)!
        return cnnFileData
    } ()
    
    /// The math of NeoML
    let mathEngine: NeoMathEngine = {
        try! NeoMathEngine.createCPUMathEngine(1)
    }()
        
    /// Classify images with NeoML
    /// - Parameters:
    ///   - data: Data (NSData) with image bytes
    ///   - completion: DocumentType from neural network output
    func classify(rawImage data: Data, completion: @escaping (DocumentType) -> ()) {
        do {
            // Create input blob for NeoML
            let inputBlob = try NeoBlob.createDnnBlob(mathEngine,
                                                       blobType: .float32,
                                                       batchLength: 1, // Put 1 image in input
                                                       batchWidth: 1, // Run 1 image at once
                                                       height: 224, // Image resolution for TextNotTextClassifier network
                                                       width: 224,  //
                                                       depth: 1,
                                                       channelCount: 3) // RGB components without alpha
            // Set image data input
            try inputBlob.setData(data)
            // Initialize the network
            let dnn = try NeoDnn.createDnn(mathEngine, data: networkFileData)
            // Fill the blob with image data
            try dnn.setInputBlob(0, blob: inputBlob)
            // Run network
            try dnn.run()
            // Get data result
            let result = try dnn.getOutputBlob(0)
            let outputData = try result.getData()
            
            // Network output is array with two floats [val1, val2] in range [0,1]
            var values = Array<Float32>(repeating: 0, count: outputData.count / MemoryLayout<Float32>.stride) // create empty array
            _ = values.withUnsafeMutableBytes { outputData.copyBytes(to: $0) } // wirte output data in array
            // Get argmax which will be recognition result
            completion(values[0] > values[1] ? DocumentType.text : DocumentType.nonText)
        } catch {
            completion(.unrecognized)
            print(error)
        }
    }
    
}

