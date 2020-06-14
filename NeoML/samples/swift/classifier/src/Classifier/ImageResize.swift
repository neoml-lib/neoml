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

extension UIImage {
    
    /// Resized image
    /// - Parameters:
    ///   - size: New size
    ///   - interpolationQuality: Interpolation quality
    func resized(to size: CGSize, interpolationQuality: CGInterpolationQuality = .medium) -> UIImage {
        
        let drawTransposed: Bool
        
        switch imageOrientation {
        case .left, .leftMirrored, .right, .rightMirrored:
            drawTransposed = true;
        default:
            drawTransposed = false
        }
        return resized(to: size, transform: orientationTransform(size: size), drawTransposed: drawTransposed, interpolationQuality: interpolationQuality)
    }
    
    private func resized(to size: CGSize, transform: CGAffineTransform, drawTransposed transpose: Bool, interpolationQuality quality: CGInterpolationQuality) -> UIImage {
        let newRect = CGRect(x: 0, y: 0, width: size.width, height: size.height).integral
        let transposedRect = CGRect(x: 0, y: 0, width: newRect.height, height: newRect.width)
        guard let imageRef = cgImage else {
            fatalError("Invalid Image")
        }
        // Build a context that's the same dimensions as the new size
        guard let bitmap = CGContext(data: nil,
                                     width: Int(newRect.width),
                                     height: Int(newRect.height),
                                     bitsPerComponent: imageRef.bitsPerComponent,
                                     bytesPerRow: 0,
                                     space: imageRef.colorSpace ?? CGColorSpaceCreateDeviceRGB(),
                                     bitmapInfo: imageRef.bitmapInfo.rawValue)
            else {
                fatalError("Invalid Image")
        }
        
        // Rotate and/or flip the image if required by its orientation
        bitmap.concatenate(transform)
        
        // Set the quality level to use when rescaling
        bitmap.interpolationQuality = quality;
        
        // Draw into the context; this scales the image
        bitmap.draw(imageRef, in: transpose ? transposedRect : newRect)
        
        // Get the resized image from the context and a UIImage
        guard let newImageRef = bitmap.makeImage() else {
            fatalError("Invalid Image")
        }
        let newImage = UIImage(cgImage: newImageRef)
        return newImage
    }
    
    private func orientationTransform(size: CGSize) -> CGAffineTransform {
        
        var transform: CGAffineTransform = .identity
        
        switch imageOrientation {
        case .down, .downMirrored: // EXIF = 3 , EXIF = 4
            transform = transform.translatedBy(x: size.width, y: size.height).rotated(by: .pi)
        case .left, .leftMirrored: // EXIF = 6 , EXIF = 5
            transform = transform.translatedBy(x: size.width, y: 0).rotated(by: .pi / 2)
        case .right, .rightMirrored: // EXIF = 8 , EXIF = 7
            transform = transform.translatedBy(x: 0, y: size.height).rotated(by: -.pi / 2)
        default:
            break
        }
        
        switch imageOrientation {
        case .upMirrored, .downMirrored: // EXIF = 2 , EXIF = 4
            transform = transform.translatedBy(x: size.width, y: 0).scaledBy(x: -1, y: 1)
        case .leftMirrored, .rightMirrored: // EXIF = 5, EXIF = 7
            transform = transform.translatedBy(x: size.height, y: 0).scaledBy(x: -1, y: 1)
        default:
            break
        }
        return transform
    }
    
}
