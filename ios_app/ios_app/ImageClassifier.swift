@preconcurrency import Vision
import CoreML
import UIKit
import Combine

@MainActor
final class ImageClassifier: ObservableObject {
    @Published var results: [String] = []
    @Published var isBusy = false

    private let visionModel: VNCoreMLModel

    init() {
        let config = MLModelConfiguration()
        // MobileNetV2 — класс, сгенерированный Xcode из .mlpackage
        guard let coreMLModel = try? MobileNetV2(configuration: config).model,
              let vnModel = try? VNCoreMLModel(for: coreMLModel) else {
            fatalError("Не удалось загрузить модель MobileNetV2")
        }
        self.visionModel = vnModel
    }

    func classify(_ image: UIImage) {
        guard let cgImage = image.cgImage else { return }
        isBusy = true
        results = []
        let model = visionModel

        // Инференс в фоне. request/handler живут только внутри runClassification
        // и не пересекают границу задачи — это то, чего требует Swift 6.
        Task.detached {
            let output = Self.runClassification(model: model, cgImage: cgImage)
            await MainActor.run {
                self.results = output
                self.isBusy = false
            }
        }
    }

    nonisolated private static func runClassification(
        model: VNCoreMLModel,
        cgImage: CGImage
    ) -> [String] {
        let request = VNCoreMLRequest(model: model)
        request.imageCropAndScaleOption = .centerCrop

        let handler = VNImageRequestHandler(cgImage: cgImage, orientation: .up)
        do {
            try handler.perform([request])
        } catch {
            return []
        }

        guard let obs = request.results as? [VNClassificationObservation] else { return [] }
        // топ-3 предсказания
        return obs.prefix(3).map { "\($0.identifier) — \(Int($0.confidence * 100))%" }
    }
}
