# CoreML iOS Demo — MobileNetV2 Image Classifier

Демо-проект к ДЗ по курсу «Разработка интеллектуальных приложений»:
конвертация модели из PyTorch в **Core ML** и её запуск в iOS-приложении
(SwiftUI + Vision).

Модель: **MobileNetV2** из `torchvision` (эталонная реализация = PyTorch Hub),
классификатор изображений ImageNet на 1000 классов.

## Структура

```
.
├── convert/
│   ├── convert.py         # MobileNetV2 (PyTorch) -> MobileNetV2.mlpackage
│   ├── convert_hf.py      # вариант: DistilBERT sentiment из Hugging Face
│   └── requirements.txt
├── ios/CoreMLDemo/
│   ├── CoreMLDemoApp.swift # точка входа
│   ├── ContentView.swift   # UI: выбор фото + вывод топ-3 классов
│   └── ImageClassifier.swift # обёртка Core ML + Vision
├── docs/                   # решение (PDF)
├── .gitignore
└── LICENSE
```

## 1. Конвертация модели

```bash
cd convert
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python convert.py            # -> MobileNetV2.mlpackage
```

> `coremltools` устойчиво работает на **Python 3.9–3.11**. Конвертация и
> запуск приложения требуют **macOS + Xcode 15+**.

## 2. iOS-приложение

1. Xcode → New Project → App → Interface: **SwiftUI**, Deployment target **iOS 17+**.
2. Заменить сгенерированные `App`/`ContentView` файлами из `ios/CoreMLDemo/`,
   добавить `ImageClassifier.swift`.
3. Перетащить `MobileNetV2.mlpackage` в навигатор проекта
   (галочка *Copy items if needed*) — Xcode сгенерирует Swift-класс `MobileNetV2`.
4. ▶ Run. Выбрать фото из галереи → приложение покажет топ-3 класса с вероятностями.

## Как это работает

Выбор фото → `UIImage` уходит в `ImageClassifier` → Vision (`VNCoreMLRequest`)
сам масштабирует картинку под вход 224×224 и запускает инференс на
Neural Engine → на экран выводятся топ-3 класса.

Нормализация ImageNet (`scale`/`bias`) вшита прямо в модель при конвертации,
поэтому на стороне Swift ручная предобработка пикселей не нужна.

## Замена модели

Сконвертировать другую модель, перетащить новый `.mlpackage`, поменять имя
класса в `ImageClassifier` (`MobileNetV2(...)` → новый класс). Логика Vision
остаётся прежней.

## Лицензия

MIT — см. [LICENSE](LICENSE).
