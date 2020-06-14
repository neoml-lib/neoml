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

//Custom colors
extension UIColor {
    
    static let appRed = UIColor(red:239, green:30, blue:38) // red
    static let appGreen = UIColor(red:3, green:154, blue:85) //green
    static let appDarkBlack = UIColor(red: 28, green: 28, blue: 28, alpha: 0.97)
    static let appWhite20 = UIColor.white.withAlphaComponent(0.2)
    static let appWhite50 = UIColor.white.withAlphaComponent(0.5)
    
    convenience init(red: Int, green: Int, blue: Int, alpha: Float = 1.0) {
        self.init(red: CGFloat(red) / 255.0, green: CGFloat(green) / 255.0, blue: CGFloat(blue) / 255.0, alpha: CGFloat(alpha))
    }
}
