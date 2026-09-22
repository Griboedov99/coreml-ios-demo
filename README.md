# CoreML iOS Demo — MobileNetV2 Image Classifier

Демо-проект к ДЗ по курсу «Разработка интеллектуальных приложений»:
конвертация модели из **PyTorch** в **Core ML** и её запуск в iOS-приложении
на SwiftUI + Vision.

Модель — **MobileNetV2** из `torchvision` (эталонная реализация = PyTorch Hub),
классификатор изображений ImageNet на 1000 классов. Приложение позволяет
выбрать фото из галереи и показывает топ-3 предсказанных класса с вероятностями.

## Структура

```
.
├── convert/
│   ├── convert.py          # MobileNetV2 (PyTorch) -> MobileNetV2.mlpackage
│   ├── convert_hf.py       # вариант: DistilBERT sentiment из Hugging Face
│   └── requirements.txt
├── ios_app/                # Xcode-проект (SwiftUI)
│   └── ios_app/
│       ├── ios_appApp.swift    # точка входа
│       ├── ContentView.swift   # UI: выбор фото + вывод топ-3 классов
│       └── ImageClassifier.swift # обёртка Core ML + Vision
├── docs/                   # отчёт по ДЗ (PDF)
├── LICENSE
└── README.md
```

## 1. Конвертация модели

Требуется **macOS** и **Python 3.9–3.11** (со свежими 3.12/3.13 `coremltools`
пока конфликтует).

```bash
cd convert
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 convert.py            # -> MobileNetV2.mlpackage
```

В результате рядом появится `MobileNetV2.mlpackage`.

## 2. Запуск iOS-приложения

1. Открыть `ios_app/ios_app.xcodeproj` в **Xcode 15+** (Deployment target iOS 17+).
2. Перетащить полученный `MobileNetV2.mlpackage` в навигатор проекта
   (галочка *Copy items if needed*) — Xcode сгенерирует Swift-класс `MobileNetV2`.
3. ▶ Run на симуляторе или устройстве.
4. Выбрать фото из галереи → приложение покажет топ-3 класса с вероятностями.

## Как это работает

Выбор фото → `UIImage` уходит в `ImageClassifier` → Vision (`VNCoreMLRequest`)
сам масштабирует картинку под вход 224×224 и запускает инференс → на экран
выводятся топ-3 класса.

Две вещи «вшиты» в модель на этапе конвертации, чтобы упростить Swift-код:

- **Нормализация ImageNet** (`scale`/`bias`) — поэтому вручную обрабатывать
  пиксели на стороне iOS не нужно.
- **Softmax** на выходе — модель отдаёт вероятности (0..1), а не «сырые» логиты,
  поэтому проценты в UI корректны.

> Сумма трёх показанных процентов обычно меньше 100% — это нормально: классов
> 1000, и оставшаяся вероятность распределена по остальным категориям.

## Замена модели

Сконвертировать другую модель, перетащить новый `.mlpackage`, поменять имя
класса в `ImageClassifier` (`MobileNetV2(...)` → новый класс). Логика Vision
остаётся прежней. Пример конвертации модели из Hugging Face — в
`convert/convert_hf.py`.

## Заметки

- Модель `.mlpackage` не хранится в репозитории (большой бинарный файл, в
  `.gitignore`) — она собирается одной командой из `convert/`.
- На симуляторе Core ML работает на CPU (нет Neural Engine), в логах возможны
  сообщения про `MPSGraph` — это не ошибка проекта, инференс отрабатывает.

## Лицензия

MIT — см. [LICENSE](LICENSE).
