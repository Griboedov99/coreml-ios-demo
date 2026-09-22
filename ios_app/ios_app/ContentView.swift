import SwiftUI
import PhotosUI

struct ContentView: View {
    @StateObject private var classifier = ImageClassifier()
    @State private var pickerItem: PhotosPickerItem?
    @State private var image: UIImage?

    var body: some View {
        VStack(spacing: 20) {
            Group {
                if let image {
                    Image(uiImage: image)
                        .resizable()
                        .scaledToFit()
                } else {
                    RoundedRectangle(cornerRadius: 12)
                        .fill(.gray.opacity(0.15))
                        .overlay(Text("Выберите изображение").foregroundStyle(.secondary))
                }
            }
            .frame(height: 300)
            .cornerRadius(12)

            PhotosPicker("Выбрать из галереи", selection: $pickerItem, matching: .images)
                .buttonStyle(.borderedProminent)

            if classifier.isBusy {
                ProgressView()
            } else {
                ForEach(classifier.results, id: \.self) { line in
                    Text(line).font(.headline)
                }
            }
            Spacer()
        }
        .padding()
        .onChange(of: pickerItem) { _, newItem in
            Task {
                guard let data = try? await newItem?.loadTransferable(type: Data.self),
                      let uiImage = UIImage(data: data) else { return }
                image = uiImage
                classifier.classify(uiImage)
            }
        }
    }
}
