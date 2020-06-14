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

/// Image Converter
class ImageConverter {
    
    /// Create Data of floats from UIImage
    /// - Parameters:
    ///   - image: input image
    ///   - completion: output completion with data
    func rawData(for image: UIImage, completion:@escaping (Data?) -> ()) {
        DispatchQueue.global(qos: .utility).async {
            // TextNotTextClassifier.dnn uses images with 224x224 resolution
            let result = self.data(from: image.resized(to: CGSize(width: 224, height: 224)))
            completion(result)
        }
    }
    
    /// Create array with RGB float components  for image and put it in Data
    /// - Parameter image: input image
    func data(from image: UIImage) -> Data? {
        
        guard let imageRef = image.cgImage else { return nil }
        
        let floatSize = MemoryLayout<Float32>.stride;
        let width = imageRef.width
        let height = imageRef.height
        let components = 3
        let bitsPerComponent = 8
        let bytesPerPixel = 4
        let totalPixels = height * width
        let totalBytes = totalPixels * bytesPerPixel
        let bytesPerRow = bytesPerPixel * width
        
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        let bitmapInfo = CGImageAlphaInfo.noneSkipLast
        
        let pixels = UnsafeMutablePointer<UInt8>.allocate(capacity: totalBytes)
        pixels.initialize(repeating: 0, count: totalBytes)
        
        // Create bitmap from UIImage
        guard let bitmap = CGContext(data: pixels,
                                     width: width,
                                     height: height,
                                     bitsPerComponent: bitsPerComponent,
                                     bytesPerRow: bytesPerRow,
                                     space: colorSpace,
                                     bitmapInfo: bitmapInfo.rawValue) else { return nil }
        
        bitmap.draw(imageRef, in: CGRect(origin: .zero, size: CGSize(width: width, height: height)))
        // Normalized RGB components array
        var resultData = Data(capacity: totalPixels * components * floatSize)
        
        for i in 0..<height {
            for j in 0..<width {
                // Next pixel index
                let index = (width * i + j) * bytesPerPixel
                // RGB components in [0, 1] range
                let red = pixels[index].normalized
                let green = pixels[index+1].normalized
                let blue = pixels[index+2].normalized
                // Put RGB components in array
                resultData.append(value: red)
                resultData.append(value: green)
                resultData.append(value: blue)
            }
        }
        return resultData
    }
}

extension Data {
    mutating func append<T>(value: T) {
        Swift.withUnsafeBytes(of: value) { buffer in
            self.append(buffer.bindMemory(to: T.self))
        }
    }
}

/// Normalize pixel RGB components to range [0, 1]
private extension UInt8 {
    var normalized: Float32 {
        return Float32(self) / 255.0;
    }
}
