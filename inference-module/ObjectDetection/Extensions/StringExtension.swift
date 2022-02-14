
import UIKit

extension String {

  /**This method gets size of a string with a particular font.
   */
  func size(usingFont font: UIFont) -> CGSize {
    return size(withAttributes: [.font: font])
  }

}

