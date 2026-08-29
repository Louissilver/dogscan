# Treino no Google Colab — DogScan (YOLO11n, 16 raças)

Máquina local não tem GPU. Colab T4 grátis treina isso em ~30-60 min.

## 1. Zipar o dataset (na máquina local)

```bash
cd D:/Projetos/vision
tar -czf dataset_dogs.tgz -C data dataset_dogs
```

Subir `dataset_dogs.tgz` + `training/data.yaml` pro Colab (ou pro Google Drive).

## 2. Notebook Colab

Runtime → Change runtime type → **T4 GPU**.

```python
# --- setup ---
!pip -q install ultralytics
from google.colab import files
up = files.upload()          # dataset_dogs.tgz  e  data.yaml
!tar -xzf dataset_dogs.tgz   # -> ./dataset_dogs/

# data.yaml precisa apontar pro path certo no Colab:
import pathlib
p = pathlib.Path('data.yaml')
p.write_text(p.read_text().replace('../data/dataset_dogs', 'dataset_dogs'))
print(p.read_text())
```

```python
# --- treino ---
from ultralytics import YOLO

model = YOLO('yolo11n.pt')
model.train(
    data='data.yaml',
    epochs=100,
    imgsz=640,
    batch=32,
    patience=25,
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
    degrees=10, translate=0.1, scale=0.5, fliplr=0.5,
    mosaic=1.0, close_mosaic=10,
)
m = model.val()
print('mAP50-95', m.box.map, 'mAP50', m.box.map50)
```

```python
# --- export TFLite int8 ---
from ultralytics import YOLO
best = YOLO('runs/detect/train/weights/best.pt')
path = best.export(format='tflite', int8=True, data='data.yaml', imgsz=640)
print(path)

from google.colab import files
files.download(path)         # baixa best_full_integer_quant.tflite (ou _int8)
```

## 3. Na máquina local

```
copiar o .tflite baixado  ->  app/assets/models/dogscan.tflite
```

Editar `app/lib/main.dart`: `const bool kUseTrainedModel = true;`

```bash
cd app && flutter run --release
```

## Notas

- Se `mAP50` < ~0.6, aumentar `epochs`, ou trocar `yolo11n.pt` por `yolo11s.pt`
  (mais lento no celular, mais preciso).
- O plugin `ultralytics_yolo` lê metadados embutidos no TFLite exportado pela
  Ultralytics (nomes das classes inclusos). Não precisa mandar `metadata.yaml`
  junto — ele existe só como referência humana.
- Nome exato do arquivo exportado varia (`best_int8.tflite`,
  `best_full_integer_quant.tflite`). Renomeia pra `dogscan.tflite`.
