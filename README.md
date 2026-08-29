<div align="center">

# 🐶 DogScan

**Real-time dog-breed identification from your phone's camera.**

Point the camera at a dog — DogScan draws a box around it and labels the breed, live and on-device.

![Flutter](https://img.shields.io/badge/Flutter-3.47-02569B?logo=flutter&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![YOLO11](https://img.shields.io/badge/model-YOLO11n-blue)
![Platform](https://img.shields.io/badge/platform-Android%20%7C%20iOS-lightgrey)
![License](https://img.shields.io/badge/license-MIT-green)
![Status](https://img.shields.io/badge/status-WIP-orange)

</div>

---

## Table of contents

- [Overview](#overview)
- [How it works](#how-it-works)
- [Supported breeds](#supported-breeds)
- [Features](#features)
- [Tech stack](#tech-stack)
- [Repository structure](#repository-structure)
- [Getting started](#getting-started)
- [The model](#the-model)
- [Configuration](#configuration)
- [Platform notes](#platform-notes)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Disclaimer](#disclaimer)

## Overview

DogScan is a Flutter app plus a small training pipeline. The app runs a YOLO object-detection
model on the live camera feed and overlays a bounding box with the predicted breed and a
confidence score. Inference runs **on-device** (TensorFlow Lite) — no server, works offline
once the model is bundled.

The repository ships:

| Piece | What it is |
|-------|------------|
| `app/` | Flutter app (`dogscan`) — live camera + detection overlay |
| `data/` | Dataset preparation: Stanford Dogs → YOLO detection format |
| `training/` | YOLO11n training + export to TFLite (run on Colab) |

A trained **16-breed** model (`app/assets/models/dogscan.tflite`, YOLO11n int8, ~3 MB) is
committed to the repo, so a fresh clone runs the breed detector offline with no extra steps.
You can also retrain it yourself — see [The model](#the-model).

## How it works

```
┌─────────────────┐   VOC boxes    ┌──────────────────┐   YOLO txt    ┌───────────────┐
│  Stanford Dogs  │ ─────────────► │ data/dogs_prepare│ ────────────► │ data/dataset_ │
│  20,580 imgs    │                │      .py          │               │   dogs/       │
└─────────────────┘                └──────────────────┘               └──────┬────────┘
                                                                            │
                                                            train (Colab, GPU)
                                                                            ▼
┌─────────────────┐   TFLite int8   ┌──────────────────┐   best.pt    ┌───────────────┐
│  Flutter app    │ ◄────────────── │ training/export  │ ◄─────────── │  YOLO11n      │
│  YOLOView       │                 │      .py          │              │  fine-tuned   │
└────────┬────────┘                 └──────────────────┘              └───────────────┘
         │ per frame
         ▼
  camera frame → YOLO inference → NMS → box + label + confidence → overlay
```

## Supported breeds

The model detects these 16 classes. Edit the `BREEDS` list in
[`data/dogs_prepare.py`](data/dogs_prepare.py) to change them, then retrain (Option B below).

| # | class id (slug) | display name (pt-BR) | Stanford Dogs folder |
|---|-----------------|----------------------|----------------------|
| 0 | `labrador` | Labrador | `Labrador_retriever` |
| 1 | `golden_retriever` | Golden Retriever | `golden_retriever` |
| 2 | `pastor_alemao` | Pastor-alemão | `German_shepherd` |
| 3 | `bulldog_frances` | Bulldog Francês | `French_bulldog` |
| 4 | `boxer` | Boxer | `boxer` |
| 5 | `beagle` | Beagle | `beagle` |
| 6 | `rottweiler` | Rottweiler | `Rottweiler` |
| 7 | `pug` | Pug | `pug` |
| 8 | `chihuahua` | Chihuahua | `Chihuahua` |
| 9 | `husky` | Husky Siberiano | `Siberian_husky` |
| 10 | `spitz_pomeranian` | Spitz Alemão (Pomerânia) | `Pomeranian` |
| 11 | `yorkshire` | Yorkshire Terrier | `Yorkshire_terrier` |
| 12 | `doberman` | Dobermann | `Doberman` |
| 13 | `border_collie` | Border Collie | `Border_collie` |
| 14 | `shih_tzu` | Shih Tzu | `Shih-Tzu` |
| 15 | `pinscher` | Pinscher Miniatura | `miniature_pinscher` |

Anything outside this list is forced into the nearest class — a Poodle will read as
something, just wrongly.

## Features

- 📷 Full-screen live camera with native bounding-box overlay
- 🧠 On-device inference (TFLite) — offline once the model is bundled
- 🏷️ Top-1 breed label with a confidence bar, localized names (pt-BR)
- 🔁 Drop-in model swap: generic COCO fallback ↔ your trained breed model via one flag
- 🛠️ Reproducible dataset + training pipeline from a public dataset

## Tech stack

| Layer | Choice |
|-------|--------|
| App | Flutter 3.47, Dart 3.13 |
| Inference | [`ultralytics_yolo`](https://pub.dev/packages/ultralytics_yolo) `^0.6.14` (TFLite / Core ML) |
| Model | Ultralytics YOLO11n, transfer-learned from COCO |
| Data | [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/) (120 breeds, VOC boxes) |
| Training | Python 3.11, `ultralytics`, Google Colab (T4) |
| Permissions | `permission_handler` |

## Repository structure

```
.
├── app/                      Flutter application
│   ├── lib/main.dart         live camera + YOLOView + overlay
│   ├── assets/models/        dogscan.tflite (16-breed model, committed)
│   └── README.md             app-specific setup & troubleshooting
├── data/
│   ├── dogs_prepare.py       Stanford Dogs → YOLO dataset + training/data.yaml
│   └── _archive_spider/      earlier spider-ID experiment (inactive)
├── training/
│   ├── data.yaml             generated by dogs_prepare.py
│   ├── train.py              YOLO11n training entrypoint
│   ├── export.py             → TFLite (Linux/macOS only)
│   └── colab_dogs.md         step-by-step Colab notebook
├── requirements-data.txt     dataset pipeline deps (light)
└── requirements-train.txt    training/export deps (ultralytics + tf)
```

## Getting started

### Prerequisites

- **Flutter** 3.47+ (`flutter doctor` green for Android)
- **Android SDK** with platform 36 + build-tools 36, licenses accepted
- **JDK 17** (Gradle for the Android build; point Flutter at it with `flutter config --jdk-dir <path>`)
- **Python** 3.11 (only for the dataset/training pipeline)
- An Android device with USB debugging (camera does not work well on emulators)

### Run the app

```bash
cd app
flutter pub get
flutter run --release
```

The 16-breed model ships in the repo, so this works offline immediately — point the camera
at a dog and you get a breed label. No download, no training step.

### Build an APK

```bash
cd app
flutter build apk --release          # build/app/outputs/flutter-apk/app-release.apk
# or smaller, per-ABI:
flutter build apk --release --split-per-abi
```

## The model

There are two ways to get a model into the app.

### Option A — use the one that ships with the repo (default)

Nothing to do. [`app/assets/models/dogscan.tflite`](app/assets/models) is a YOLO11n int8
detector for the 16 breeds listed above, trained on a Stanford Dogs subset (~2,750 images).
`kUseTrainedModel` is already `true`, so `flutter run` just works. Accuracy is rough —
it is a nano model on a small dataset and confuses visually similar breeds.

### Option B — train your own

Retrain with different breeds, more data, or a bigger backbone. Full walkthrough:
[`training/colab_dogs.md`](training/colab_dogs.md). Summary:

**1. Prepare the dataset** (local, one-time)

```bash
# download & extract Stanford Dogs images.tar + annotation.tar somewhere, then:
pip install -r requirements-data.txt
python data/dogs_prepare.py --src /path/to/stanford --out data/dataset_dogs
tar -czf dataset_dogs.tgz -C data dataset_dogs
```

**2. Train on Colab** (GPU)

```python
!pip install ultralytics
from ultralytics import YOLO
YOLO('yolo11n.pt').train(data='data.yaml', epochs=100, imgsz=640, batch=32)
```

**3. Export to TFLite** — on Colab (Ultralytics blocks TFLite export on Windows)

```python
YOLO('runs/detect/train/weights/best.pt').export(format='tflite', int8=True, data='data.yaml')
```

**4. Wire it into the app**

```bash
cp best_int8.tflite app/assets/models/dogscan.tflite   # replace the bundled one
# kUseTrainedModel is already true in app/lib/main.dart
flutter build apk --release
```

If you change the breed list, also update `kBreedPt` in `app/lib/main.dart` and
`app/assets/models/metadata.yaml`.

## Configuration

Constants in [`app/lib/main.dart`](app/lib/main.dart):

| Constant | Default | Effect |
|----------|---------|--------|
| `kUseTrainedModel` | `true` | `true` = bundled `assets/models/dogscan.tflite` (16 breeds); `false` = generic COCO `yolo26n` (needs Wi-Fi once, detects `dog` only) |
| `kConfidence` | `0.5` | Detection confidence threshold |
| `kBreedPt` | map | Slug → display name (pt-BR) |

## Platform notes

Windows-specific tweaks already applied in `app/android/`:

- `gradle.properties`: `kotlin.incremental=false` — the Kotlin incremental cache fails to
  write its `.tab` files under Windows file locking.
- `app/build.gradle.kts`: `isMinifyEnabled = false` for `release` — R8 was stripping
  `androidx.work` / Room classes that `ultralytics_yolo` needs, crashing on launch.
- `AndroidManifest.xml`: `INTERNET` permission added — Flutter's release manifest omits it,
  but the plugin needs it to fetch models by id.

**Xiaomi / MIUI devices:** enable *USB debugging*, *Install via USB* and *USB debugging
(security settings)* in Developer options. If `adb install` still returns
`INSTALL_FAILED_USER_RESTRICTED`, push the APK and install it from the Files app:

```bash
adb push app/build/app/outputs/flutter-apk/app-release.apk /sdcard/Download/dogscan.apk
```

**TFLite export** only runs on Linux x86 / macOS — use Colab.

## Roadmap

- [x] Train and bundle the 16-breed YOLO11n model (runs offline on a fresh clone)
- [ ] Temporal smoothing of boxes between frames
- [ ] Publish `best.pt` + full training run as a GitHub Release
- [ ] Expand breed list; compare YOLO11n vs YOLO11s (mAP vs FPS)
- [ ] iOS build + Core ML export
- [ ] Release signing config + CI build

## Contributing

Issues and PRs welcome. For model/data changes, keep `data/dogs_prepare.py` as the single
source of truth for the class list and regenerate `training/data.yaml` from it.

## License

[MIT](LICENSE) for the code in this repository.

The **Stanford Dogs Dataset** is released for **non-commercial research use** and is *not*
redistributed here — you download it yourself. Trained weights derived from it inherit that
restriction. Review the dataset terms before any commercial use.

## Acknowledgements

- [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/) — Khosla et al.
- [Ultralytics YOLO](https://docs.ultralytics.com) and the `ultralytics_yolo` Flutter plugin

## Disclaimer

DogScan is a toy. Predictions come from a small model and are frequently wrong, especially
between visually similar breeds. It is not a substitute for a breeder, a vet, or a DNA test.
