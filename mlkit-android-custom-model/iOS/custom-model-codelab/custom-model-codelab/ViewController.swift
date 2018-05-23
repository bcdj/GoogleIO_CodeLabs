//
//  ViewController.swift
//  custom-model-codelab
//
//  Created by Chip Snyder on 5/20/18.
//  Copyright © 2018 EisbarDev. All rights reserved.
//

import UIKit
import Firebase

struct Probablity {
    let label : String
    let probability : Float
}

class ViewController: UIViewController {
    
    @IBOutlet weak var imageView: UIImageView!
    @IBOutlet weak var imageToggle: UISwitch!
    @IBOutlet weak var textView: UITextView!
    @IBOutlet weak var runModel: UIButton!
    
    private let HOSTED_MODEL_NAME = "mobilenet_v1_224_quant"
    private let LOCAL_MODEL_NAME = "mobilenet_v1_1.0_224_quant"
    private let TENNIS_IMAGE_NAME = "tennis"
    private let MOUNTAIN_IMAGE_NAME = "mountain"
    private let LABELS_FILE = "labels"
    private let DIM_BATCH_SIZE = NSNumber(value:Float(1));
    private let DIM_PIXEL_SIZE = NSNumber(value:Float(3));
    private let DIM_IMG_SIZE_X = NSNumber(value:Float(224));
    private let DIM_IMG_SIZE_Y = NSNumber(value:Float(224));
    
    private let labelsList : [String]
    private var interpreter : ModelInterpreter?
    private var ioOptions : ModelInputOutputOptions?
    
    required init?(coder aDecoder: NSCoder) {
        let text : String
        do {
            if let url = Bundle.main.url(forResource:LABELS_FILE, withExtension: "txt") {
                text = try String(contentsOf: url).trimmingCharacters(in: .whitespacesAndNewlines)
            } else {
                text = ""
            }
        } catch {
            text = ""
        }
        labelsList = text.components(separatedBy: "\n")
        
        super.init(coder: aDecoder)
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        let conditions = ModelDownloadConditions(wiFiRequired: true, idleRequired: false)
        let cloudModelSource = CloudModelSource(
            modelName: HOSTED_MODEL_NAME,
            enableModelUpdates: true,
            initialConditions: conditions,
            updateConditions: conditions
        )
        
        ModelManager.modelManager().register(cloudModelSource)
        
        if let modelPath = Bundle.main.path(forResource: LOCAL_MODEL_NAME, ofType: "tflite") {
            let localModelSource = LocalModelSource(modelName: LOCAL_MODEL_NAME, path: modelPath)
            ModelManager.modelManager().register(localModelSource)
        }
        
        let options : ModelOptions
        options = ModelOptions(
            cloudModelName: HOSTED_MODEL_NAME,
            localModelName: LOCAL_MODEL_NAME
        )
        
        interpreter = ModelInterpreter(options: options)
        
        ioOptions = ModelInputOutputOptions()
        
        if let ioOptions = ioOptions {
            do {
                try ioOptions.setInputFormat(index: 0, type: .uInt8, dimensions: [DIM_BATCH_SIZE, DIM_IMG_SIZE_X, DIM_IMG_SIZE_Y, DIM_PIXEL_SIZE])
                try ioOptions.setOutputFormat(index: 0, type: .uInt8, dimensions: [DIM_BATCH_SIZE, NSNumber(value:labelsList.count)])
            } catch let error as NSError {
                NSLog("Failed to set input or output format with error: \(error.localizedDescription)")
            }
        }
    }
    
    @IBAction func imageChanged(_ sender: UISwitch) {
        imageView.image = imageForSwitch(sender)
        textView.text = "Select run model for predictions"
    }
    
    func imageForSwitch(_ sender:UISwitch) -> UIImage? {
        let imageName = sender.isOn ? TENNIS_IMAGE_NAME : MOUNTAIN_IMAGE_NAME
        return UIImage(named:imageName)
    }
    
    func scaledImageData(_ image:UIImage) -> Data? {
        guard let cgImage = image.cgImage, cgImage.width > 0 else { return nil }
        let oldComponentsCount = cgImage.bytesPerRow / cgImage.width
        guard DIM_PIXEL_SIZE.intValue <= oldComponentsCount else { return nil }

        let newWidth = DIM_IMG_SIZE_X.intValue
        let newHeight = DIM_IMG_SIZE_Y.intValue
        let dataSize = newWidth * newHeight * oldComponentsCount
        var imageData = [UInt8](repeating: 0, count: dataSize)
        guard let context = CGContext(
            data: &imageData,
            width: newWidth,
            height: newHeight,
            bitsPerComponent: cgImage.bitsPerComponent,
            bytesPerRow: oldComponentsCount * newWidth,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGBitmapInfo.byteOrder32Big.rawValue | CGImageAlphaInfo.premultipliedLast.rawValue
            ) else {
                return nil
        }
        context.draw(cgImage, in: CGRect(x: 0, y: 0, width: newWidth, height: newHeight))
        let count = newWidth * newHeight * DIM_PIXEL_SIZE.intValue * DIM_BATCH_SIZE.intValue
        var scaledImageDataArray = [UInt8](repeating: 0, count: count)
        var pixelIndex = 0
        for _ in 0..<newWidth {
            for _ in 0..<newHeight {
                let pixel = imageData[pixelIndex]
                pixelIndex += 1

                // Ignore the alpha component.
                let red = (pixel >> 16) & 0xFF
                let green = (pixel >> 8) & 0xFF
                let blue = (pixel >> 0) & 0xFF
                scaledImageDataArray[pixelIndex] = red
                scaledImageDataArray[pixelIndex + 1] = green
                scaledImageDataArray[pixelIndex + 2] = blue
            }
        }
        let scaledImageData = Data(bytes: scaledImageDataArray)
        let newImage = UIImage(data: scaledImageData)
        
        
        return scaledImageData
    }
    
    @IBAction func runModel(_ sender: Any) {
        
        if let image = imageForSwitch(imageToggle) {
//            DispatchQueue.global(qos:.background).async {
                self.runModel(onImage: image) { labels in
//                    DispatchQueue.main.async {
                        self.textView.text = labels.joined(separator: "\n")
//                    }
                }
//            }
        }
    }
    
    func runModel(onImage image:UIImage, completion: @escaping ([String]) -> Void) {
        let input = ModelInputs()
        let data = scaledImageData(image)
        
        do {
            try input.addInput(data as Any)
        } catch let error as NSError {
            NSLog("Failed to add input: \(error.localizedDescription)")
            return
        }
        
        if let interpreter = interpreter, let ioOptions = ioOptions {
            interpreter.run(inputs: input, options: ioOptions) { (outputs, error) in
                if let error = error {
                    print("\(error.localizedDescription)")
                }
                
                let probabilities = try? outputs?.output(index: 0)
                
                if let probabilitiesArray = probabilities as? [[UInt8]] {
                   completion(self.topLabels(byteArray: probabilitiesArray[0]))
                }
                
            }
        }
    }
    
    func topLabels(byteArray:[UInt8]) -> [String] {
        
        var labels = Array<Probablity>()
        
        for (index, element) in byteArray.enumerated() {
            let probaility = Float(Float((element & 0xff))/255.0)
            labels.append(Probablity(label: labelsList[index], probability: probaility))
        }
        
        labels.sort(by: { $0.probability > $1.probability })
        print(labels)
        
        var topLabels = Array<String>()
        for index in 0...3 {
            topLabels.append("\(labels[index].label) : \(labels[index].probability)")
        }
        
        return topLabels
    }
}
