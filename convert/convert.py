"""
convert.py — конвертация MobileNetV2 (PyTorch / torchvision) в Core ML.

Результат: MobileNetV2.mlpackage — классификатор изображений ImageNet
(1000 классов), готовый к перетаскиванию в Xcode.

Запуск:
    python convert.py
"""
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import torchvision
import coremltools as ct

# 1. Предобученная модель (эталонная реализация из torchvision = PyTorch Hub)
weights = torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V2
base_model = torchvision.models.mobilenet_v2(weights=weights)

# MobileNetV2 отдаёт "сырые" логиты (без нормализации). Добавляем Softmax,
# чтобы на выходе были вероятности 0..1 — иначе Vision вернёт confidence
# больше 1 и в UI получатся проценты вроде 639%.
torch_model = nn.Sequential(base_model, nn.Softmax(dim=1))
torch_model.eval()

# 2. Трассировка с примером входа нужной формы (NCHW, 224x224)
example_input = torch.rand(1, 3, 224, 224)
traced_model = torch.jit.trace(torch_model, example_input)

# 3. Метки классов ImageNet (1000 шт.)
label_url = "https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt"
class_labels = urllib.request.urlopen(label_url).read().decode("utf-8").splitlines()
class_labels = class_labels[1:]          # первый класс — background, убираем
assert len(class_labels) == 1000

# 4. Предобработка ImageNet "вшивается" в модель.
#    ImageType применяет: pixel * scale + bias (bias — по каналам).
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
scale = 1.0 / (0.226 * 255.0)            # усреднённый std (документация Apple)
bias = (-mean / std).tolist()

image_input = ct.ImageType(
    name="image",
    shape=example_input.shape,
    scale=scale,
    bias=bias,
)

# 5. Конвертация через Unified Conversion API
mlmodel = ct.convert(
    traced_model,
    inputs=[image_input],
    classifier_config=ct.ClassifierConfig(class_labels),
    minimum_deployment_target=ct.target.iOS16,
    compute_units=ct.ComputeUnit.ALL,     # CPU + GPU + Neural Engine
)

# 6. Метаданные — их видно в Xcode при клике на модель
mlmodel.author = "Nick"
mlmodel.short_description = "MobileNetV2 image classifier (ImageNet, 1000 classes)"
mlmodel.input_description["image"] = "Входное изображение 224x224"

# 7. Сохранение. Современный формат — .mlpackage.
#    Для legacy .mlmodel добавить в convert(): convert_to="neuralnetwork"
mlmodel.save("MobileNetV2.mlpackage")
print("Готово: MobileNetV2.mlpackage")