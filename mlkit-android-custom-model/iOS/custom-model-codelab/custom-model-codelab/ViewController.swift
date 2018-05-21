//
//  ViewController.swift
//  custom-model-codelab
//
//  Created by Chip Snyder on 5/20/18.
//  Copyright © 2018 EisbarDev. All rights reserved.
//

import UIKit
import Firebase

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
    private let DIM_BATCH_SIZE = NSNumber(value:1);
    private let DIM_PIXEL_SIZE = NSNumber(value:3);
    private let DIM_IMG_SIZE_X = NSNumber(value:224);
    private let DIM_IMG_SIZE_Y = NSNumber(value:224);
    
    private let labelsList : [String]
    private var interpreter : ModelInterpreter?
    private var ioOptions : ModelInputOutputOptions?
    
    required init?(coder aDecoder: NSCoder) {
        let text : String
        do {
            text = try String(contentsOfFile: LABELS_FILE, encoding: .utf8)
        } catch {
            text = ""
        }
        labelsList = text.components(separatedBy: "\n")
        
        super.init(coder: aDecoder)
    }
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        let conditions = ModelDownloadConditions(wiFiRequired: true, idleRequired: true)
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
        
        let options = ModelOptions(
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
    }
    
    func imageForSwitch(_ sender:UISwitch) -> UIImage? {
        let imageName = sender.isOn ? TENNIS_IMAGE_NAME : MOUNTAIN_IMAGE_NAME
        return UIImage(named:imageName)
    }
    
    @IBAction func runModel(_ sender: Any) {
        let input = ModelInputs()
        var data : Data?
        if let image = imageForSwitch(imageToggle) {
            data = UIImageJPEGRepresentation(image, 1)
        }
        
        do {
            try input.addInput(data as Any)
        } catch let error as NSError {
            NSLog("Failed to add input: \(error.localizedDescription)")
            return
        }
        
        if let interpreter = interpreter, let ioOptions = ioOptions {
            interpreter.run(inputs: input, options: ioOptions) { (outputs, _) in
                let probabilities = try? outputs?.output(index: 0)
               
            }
        }
    }
}

