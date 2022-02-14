
import CoreImage
import TensorFlowLite
import UIKit
import Accelerate

/// Stores results for a particular frame that was successfully run through the `Interpreter`.
struct Result {
    let inferenceTime: Double
    let inferences: [Inference]
}

/// for nms formatted prediction.
struct Prediction {
    let classIndex: Int
    let score: Float
    let rect: CGRect
}

/// Stores one formatted inference.
struct Inference {
    let confidence: Float
    let className: String
    let rect: CGRect
    let displayColor: UIColor
}

/// Information about a model file or labels file.
typealias FileInfo = (name: String, extension: String)

/// Information about the MobileNet SSD model.
enum Card {
    static let modelInfo: FileInfo = (name: "card", extension: "tflite")
    static let labelsInfo: FileInfo = (name: "card", extension: "txt")
}

/// This class handles all data preprocessing and makes calls to run inference on a given frame
/// by invoking the `Interpreter`. It then formats the inferences obtained and returns the top N
/// results for a successful inference.
class ModelDataHandler: NSObject {
    
    // MARK: - Internal Properties
    /// The current thread count used by the TensorFlow Lite Interpreter.
    let threadCount = 4
    
    let threshold: Float = 0.5
    
    // MARK: Model parameters
    let batchSize = 1
    let inputChannels = 3
    let inputWidth = 640
    let inputHeight = 640
    let outputRow = 25200 // YOLOv5 input size 640 * 640 output Row values 25200
    let outputColumn = 20 // class count + 5 (left, top, right, bottom, score)
    let nmsLimit = 100
    
    // image mean and std for floating model, should be consistent with parameters used in model training
    let imageMean: Float = 127.5
    let imageStd:  Float = 127.5
    
    // MARK: Private properties
    private var labels: [String] = []
    
    /// TensorFlow Lite `Interpreter` object for performing inference on a given model.
    private var interpreter: Interpreter
    
    private let bgraPixel = (channels: 4, alphaComponent: 3, lastBgrComponent: 2)
    private let rgbPixelChannels = 3
    private let colorStrideValue = 10
    private let colors = [
        UIColor.red,
        UIColor(displayP3Red: 90.0/255.0, green: 200.0/255.0, blue: 250.0/255.0, alpha: 1.0),
        UIColor.green,
        UIColor.orange,
        UIColor.blue,
        UIColor.purple,
        UIColor.magenta,
        UIColor.yellow,
        UIColor.cyan,
        UIColor.brown
    ]
    
    // MARK: - Initialization
    
    /// A failable initializer for `ModelDataHandler`. A new instance is created if the model and
    /// labels files are successfully loaded from the app's main bundle. Default `threadCount` is 1.
    init?(modelFileInfo: FileInfo, labelsFileInfo: FileInfo) {
        let modelFilename = modelFileInfo.name
        
        // Construct the path to the model file.
        guard let modelPath = Bundle.main.path(
            forResource: modelFilename,
            ofType: modelFileInfo.extension
        ) else {
            print("Failed to load the model file with name: \(modelFilename).")
            return nil
        }
        
        // Specify the options for the `Interpreter`.
        var options = Interpreter.Options()
        options.threadCount = threadCount
        do {
            // Create the `Interpreter`.
            interpreter = try Interpreter(modelPath: modelPath, options: options)
            // Allocate memory for the model's input `Tensor`s.
            try interpreter.allocateTensors()
        } catch let error {
            print("Failed to create the interpreter with error: \(error.localizedDescription)")
            return nil
        }
        
        super.init()
        
        // Load the classes listed in the labels file.
        loadLabels(fileInfo: labelsFileInfo)
    }
    
    /// This class handles all data preprocessing and makes calls to run inference on a given frame
    /// through the `Interpreter`. It then formats the inferences obtained and returns the top N
    /// results for a successful inference.
    func runModel(onFrame pixelBuffer: CVPixelBuffer) -> Result? {
        let imageWidth = CVPixelBufferGetWidth(pixelBuffer)
        let imageHeight = CVPixelBufferGetHeight(pixelBuffer)
        let sourcePixelFormat = CVPixelBufferGetPixelFormatType(pixelBuffer)
        assert(sourcePixelFormat == kCVPixelFormatType_32ARGB ||
               sourcePixelFormat == kCVPixelFormatType_32BGRA ||
               sourcePixelFormat == kCVPixelFormatType_32RGBA)
        
        
        let imageChannels = 4
        assert(imageChannels >= inputChannels)
        
        // Crops the image to the biggest square in the center and scales it down to model dimensions.
        let scaledSize = CGSize(width: inputWidth, height: inputHeight)
        guard let scaledPixelBuffer = pixelBuffer.resized(to: scaledSize) else {
            return nil
        }
        
        let interval: TimeInterval
        let output: Tensor
        do {
            let inputTensor = try interpreter.input(at: 0)
            
            // Remove the alpha component from the image buffer to get the RGB data.
            guard let rgbData = rgbDataFromBuffer(
                scaledPixelBuffer,
                byteCount: batchSize * inputWidth * inputHeight * inputChannels,
                isModelQuantized: inputTensor.dataType == .uInt8
            ) else {
                print("Failed to convert the image buffer to RGB data.")
                return nil
            }
            
            // Copy the RGB data to the input `Tensor`.
            try interpreter.copy(rgbData, toInputAt: 0)
            
            // Run inference by invoking the `Interpreter`.
            let startDate = Date()
            try interpreter.invoke()
            interval = Date().timeIntervalSince(startDate) * 1000
            
            output = try interpreter.output(at: 0)
        } catch let error {
            print("Failed to invoke the interpreter with error: \(error.localizedDescription)")
            return nil
        }
        
        // Formats the results
        let resultArray = formatResults(
            outputs: ([Float](unsafeData: output.data) ?? []) as [NSNumber],
            width: CGFloat(imageWidth),
            height: CGFloat(imageHeight)
        )
        
        // Returns the inference time and inferences
        let result = Result(inferenceTime: interval, inferences: resultArray)
        return result
    }
    
    /// Filters out all the results with confidence score < threshold and returns the top N results
    /// sorted in descending order.
    func formatResults(outputs: [NSNumber], width: CGFloat, height: CGFloat) -> [Inference]{
        var resultsArray: [Inference] = []
        
        var predictions = [Prediction]()
        for i in 0..<outputRow {
            if Float(truncating: outputs[i*outputColumn+4]) > threshold {
                let x = Double(truncating: outputs[i*outputColumn])
                let y = Double(truncating: outputs[i*outputColumn+1])
                let w = Double(truncating: outputs[i*outputColumn+2])
                let h = Double(truncating: outputs[i*outputColumn+3])
                
                let left = x - w/2
                let top = y - h/2
                let right = x + w/2
                let bottom = y + h/2
                
                var max = Double(truncating: outputs[i*outputColumn+5])
                var cls = 0
                for j in 0 ..< outputColumn-5 {
                    if Double(truncating: outputs[i*outputColumn+5+j]) > max {
                        max = Double(truncating: outputs[i*outputColumn+5+j])
                        cls = j
                    }
                }
                
                let rect = CGRect(x: left, y: top, width: right-left, height: bottom-top)
                
                let prediction = Prediction(classIndex: cls, score: Float(truncating: outputs[i*outputColumn+4]), rect: rect)
                predictions.append(prediction)
            }
        }
        
        let nms = nonMaxSuppression(boxes: predictions, limit: nmsLimit, threshold: threshold)


        if (nms.count == 0) {
            return resultsArray
        }
        for i in 0...nms.count - 1 {

            let score = nms[i].score

            // Filters results with confidence < threshold.
            guard score >= threshold else {
                continue
            }

            // Gets the output class names for detected classes from labels list.
            let outputClassIndex = Int(nms[i].classIndex)
            let outputClass = labels[outputClassIndex + 1]

            let rect = nms[i].rect

            // The detected corners are for model dimensions. So we scale the rect with respect to the
            // actual image dimensions.
            let newRect = rect.applying(CGAffineTransform(scaleX: width, y: height))

            // Gets the color assigned for the class
            let colorToAssign = colorForClass(withIndex: outputClassIndex + 1)
            let inference = Inference(confidence: score,
                                      className: outputClass,
                                      rect: newRect,
                                      displayColor: colorToAssign)
            resultsArray.append(inference)
        }

        // Sort results in descending order of confidence.
        resultsArray.sort { (first, second) -> Bool in
            return first.confidence  > second.confidence
        }
        
        return resultsArray
    }
    
    
    // The two methods nonMaxSuppression and IOU below are from  https://github.com/hollance/YOLO-CoreML-MPSNNGraph/blob/master/Common/Helpers.swift
    /**
     Removes bounding boxes that overlap too much with other boxes that have
     a higher score.
     - Parameters:
     - boxes: an array of bounding boxes and their scores
     - limit: the maximum number of boxes that will be selected
     - threshold: used to decide whether boxes overlap too much
     */
    func nonMaxSuppression(boxes: [Prediction], limit: Int, threshold: Float) -> [Prediction] {
        // Do an argsort on the confidence scores, from high to low.
        let sortedIndices = boxes.indices.sorted { boxes[$0].score > boxes[$1].score }
        
        var selected: [Prediction] = []
        var active = [Bool](repeating: true, count: boxes.count)
        var numActive = active.count
        
        // The algorithm is simple: Start with the box that has the highest score.
        // Remove any remaining boxes that overlap it more than the given threshold
        // amount. If there are any boxes left (i.e. these did not overlap with any
        // previous boxes), then repeat this procedure, until no more boxes remain
        // or the limit has been reached.
    outer: for i in 0..<boxes.count {
        if active[i] {
            let boxA = boxes[sortedIndices[i]]
            selected.append(boxA)
            if selected.count >= limit { break }
            
            for j in i+1..<boxes.count {
                if active[j] {
                    let boxB = boxes[sortedIndices[j]]
                    if IOU(a: boxA.rect, b: boxB.rect) > threshold {
                        active[j] = false
                        numActive -= 1
                        if numActive <= 0 { break outer }
                    }
                }
            }
        }
    }
        return selected
    }
    
    /**
     Computes intersection-over-union overlap between two bounding boxes.
     */
    func IOU(a: CGRect, b: CGRect) -> Float {
        let areaA = a.width * a.height
        if areaA <= 0 { return 0 }
        
        let areaB = b.width * b.height
        if areaB <= 0 { return 0 }
        
        let intersectionMinX = max(a.minX, b.minX)
        let intersectionMinY = max(a.minY, b.minY)
        let intersectionMaxX = min(a.maxX, b.maxX)
        let intersectionMaxY = min(a.maxY, b.maxY)
        let intersectionArea = max(intersectionMaxY - intersectionMinY, 0) *
        max(intersectionMaxX - intersectionMinX, 0)
        return Float(intersectionArea / (areaA + areaB - intersectionArea))
    }
    
    /// Loads the labels from the labels file and stores them in the `labels` property.
    private func loadLabels(fileInfo: FileInfo) {
        let filename = fileInfo.name
        let fileExtension = fileInfo.extension
        guard let fileURL = Bundle.main.url(forResource: filename, withExtension: fileExtension) else {
            fatalError("Labels file not found in bundle. Please add a labels file with name " +
                       "\(filename).\(fileExtension) and try again.")
        }
        do {
            let contents = try String(contentsOf: fileURL, encoding: .utf8)
            labels = contents.components(separatedBy: .newlines)
        } catch {
            fatalError("Labels file named \(filename).\(fileExtension) cannot be read. Please add a " +
                       "valid labels file and try again.")
        }
    }
    
    /// Returns the RGB data representation of the given image buffer with the specified `byteCount`.
    ///
    /// - Parameters
    ///   - buffer: The BGRA pixel buffer to convert to RGB data.
    ///   - byteCount: The expected byte count for the RGB data calculated using the values that the
    ///       model was trained on: `batchSize * imageWidth * imageHeight * componentsCount`.
    ///   - isModelQuantized: Whether the model is quantized (i.e. fixed point values rather than
    ///       floating point values).
    /// - Returns: The RGB data representation of the image buffer or `nil` if the buffer could not be
    ///     converted.
    private func rgbDataFromBuffer(
        _ buffer: CVPixelBuffer,
        byteCount: Int,
        isModelQuantized: Bool
    ) -> Data? {
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer {
            CVPixelBufferUnlockBaseAddress(buffer, .readOnly)
        }
        guard let sourceData = CVPixelBufferGetBaseAddress(buffer) else {
            return nil
        }
        
        let width = CVPixelBufferGetWidth(buffer)
        let height = CVPixelBufferGetHeight(buffer)
        let sourceBytesPerRow = CVPixelBufferGetBytesPerRow(buffer)
        let destinationChannelCount = 3
        let destinationBytesPerRow = destinationChannelCount * width
        
        var sourceBuffer = vImage_Buffer(data: sourceData,
                                         height: vImagePixelCount(height),
                                         width: vImagePixelCount(width),
                                         rowBytes: sourceBytesPerRow)
        
        guard let destinationData = malloc(height * destinationBytesPerRow) else {
            print("Error: out of memory")
            return nil
        }
        
        defer {
            free(destinationData)
        }
        
        var destinationBuffer = vImage_Buffer(data: destinationData,
                                              height: vImagePixelCount(height),
                                              width: vImagePixelCount(width),
                                              rowBytes: destinationBytesPerRow)
        
        if (CVPixelBufferGetPixelFormatType(buffer) == kCVPixelFormatType_32BGRA){
            vImageConvert_BGRA8888toRGB888(&sourceBuffer, &destinationBuffer, UInt32(kvImageNoFlags))
        } else if (CVPixelBufferGetPixelFormatType(buffer) == kCVPixelFormatType_32ARGB) {
            vImageConvert_ARGB8888toRGB888(&sourceBuffer, &destinationBuffer, UInt32(kvImageNoFlags))
        }
        
        let byteData = Data(bytes: destinationBuffer.data, count: destinationBuffer.rowBytes * height)
        if isModelQuantized {
            return byteData
        }
        
        // Not quantized, convert to floats
        let bytes = Array<UInt8>(unsafeData: byteData)!
        var floats = [Float]()
        for i in 0..<bytes.count {
            floats.append((Float(bytes[i]) - imageMean) / imageStd)
        }
        return Data(copyingBufferOf: floats)
    }
    
    /// This assigns color for a particular class.
    private func colorForClass(withIndex index: Int) -> UIColor {
        
        // We have a set of colors and the depending upon a stride, it assigns variations to of the base
        // colors to each object based on its index.
        let baseColor = colors[index % colors.count]
        
        var colorToAssign = baseColor
        
        let percentage = CGFloat((colorStrideValue / 2 - index / colors.count) * colorStrideValue)
        
        if let modifiedColor = baseColor.getModified(byPercentage: percentage) {
            colorToAssign = modifiedColor
        }
        
        return colorToAssign
    }
}

// MARK: - Extensions

extension Data {
    /// Creates a new buffer by copying the buffer pointer of the given array.
    ///
    /// - Warning: The given array's element type `T` must be trivial in that it can be copied bit
    ///     for bit with no indirection or reference-counting operations; otherwise, reinterpreting
    ///     data from the resulting buffer has undefined behavior.
    /// - Parameter array: An array with elements of type `T`.
    init<T>(copyingBufferOf array: [T]) {
        self = array.withUnsafeBufferPointer(Data.init)
    }
}

extension Array {
    /// Creates a new array from the bytes of the given unsafe data.
    ///
    /// - Warning: The array's `Element` type must be trivial in that it can be copied bit for bit
    ///     with no indirection or reference-counting operations; otherwise, copying the raw bytes in
    ///     the `unsafeData`'s buffer to a new array returns an unsafe copy.
    /// - Note: Returns `nil` if `unsafeData.count` is not a multiple of
    ///     `MemoryLayout<Element>.stride`.
    /// - Parameter unsafeData: The data containing the bytes to turn into an array.
    init?(unsafeData: Data) {
        guard unsafeData.count % MemoryLayout<Element>.stride == 0 else { return nil }
#if swift(>=5.0)
        self = unsafeData.withUnsafeBytes { .init($0.bindMemory(to: Element.self)) }
#else
        self = unsafeData.withUnsafeBytes {
            .init(UnsafeBufferPointer<Element>(
                start: $0,
                count: unsafeData.count / MemoryLayout<Element>.stride
            ))
        }
#endif  // swift(>=5.0)
    }
}
