# app/ — DogScan (Flutter)

Câmera ao vivo + detecção de raça com caixa. Plugin `ultralytics_yolo ^0.6.14`.

## Rodar

Celular Android com **depuração USB** ligada, conectado:

```bash
cd D:/Projetos/vision/app
flutter devices          # confirmar que o celular aparece
flutter run --release
```

APK standalone:

```bash
flutter build apk --release
# -> build/app/outputs/flutter-apk/app-release.apk
```

Emulador funciona mas a "câmera" é webcam/imagem falsa — ruim pra testar detecção.

## Modelo

- **Padrão**: `assets/models/dogscan.tflite` (16 raças, YOLO11n int8) já vem no repo e
  `kUseTrainedModel = true`. `flutter run` roda o detector de raças offline, sem passos extras.
- **Modelo próprio**: substituir `assets/models/dogscan.tflite` pelo seu `.tflite` e
  `flutter run`. Se mudar as raças, ajustar `kBreedPt` em [`lib/main.dart`](lib/main.dart) e
  `assets/models/metadata.yaml`.
- **Fallback COCO**: `kUseTrainedModel = false` → usa `yolo26n` (baixado on-demand, precisa
  Wi-Fi uma vez), detecta só `dog`. Útil pra isolar problema de câmera/overlay.

Treinar o `.tflite`: ver [`../training/colab_dogs.md`](../training/colab_dogs.md).

## O que já está configurado

- `pubspec.yaml`: `ultralytics_yolo`, `permission_handler`, asset dir `assets/models/`
- `android/app/build.gradle.kts`: `minSdk = 24` (exigência do plugin)
- `AndroidManifest.xml`: permissão `CAMERA`
- `ios/Runner/Info.plist`: `NSCameraUsageDescription`
- `lib/main.dart`: `YOLOView` fullscreen, pega top-1 por confiança, mapeia slug → nome PT-BR, barra de confiança no rodapé

## Ajustes rápidos (lib/main.dart)

| const | efeito |
|---|---|
| `kConfidence` | limiar de confiança (0.5). Subir = menos falso-positivo, some mais |
| `kUseTrainedModel` | `true` = modelo de raças bundlado; `false` = COCO `yolo26n` |
| `kBreedPt` | nomes de exibição das raças |

## Problemas comuns

- **Gradle/Java**: precisa JDK 17. Já apontado via `flutter config --jdk-dir`. Se reclamar:
  `flutter config --jdk-dir "D:\Projetos\toolchain\jdk17\jdk-17.0.20.1+1"`
- **`minSdkVersion`**: se algum plugin pedir mais que 24, subir em `build.gradle.kts`.
- **Primeira build** baixa Gradle + dependências (~vários min).
