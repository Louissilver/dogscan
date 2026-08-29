import 'package:flutter/material.dart';
import 'package:permission_handler/permission_handler.dart';
import 'package:ultralytics_yolo/ultralytics_yolo.dart';

void main() => runApp(const DogScanApp());

/// Troque para true depois de treinar e colocar o .tflite em
/// assets/models/dogscan.tflite (ver README na raiz).
const bool kUseTrainedModel = false;

/// Modelo COCO oficial (baixado on-demand pelo plugin) so detecta "dog"
/// generico. O modelo treinado da a raca.
const String kCocoModel = 'yolo26n';
const String kTrainedModel = 'assets/models/dogscan.tflite';

const double kConfidence = 0.5;

/// slug do modelo treinado -> nome de exibicao PT-BR
const Map<String, String> kBreedPt = {
  'labrador': 'Labrador',
  'golden_retriever': 'Golden Retriever',
  'pastor_alemao': 'Pastor-alemão',
  'bulldog_frances': 'Bulldog Francês',
  'boxer': 'Boxer',
  'beagle': 'Beagle',
  'rottweiler': 'Rottweiler',
  'pug': 'Pug',
  'chihuahua': 'Chihuahua',
  'husky': 'Husky Siberiano',
  'spitz_pomeranian': 'Spitz Alemão (Pomerânia)',
  'yorkshire': 'Yorkshire Terrier',
  'doberman': 'Dobermann',
  'border_collie': 'Border Collie',
  'shih_tzu': 'Shih Tzu',
  'pinscher': 'Pinscher Miniatura',
  // fallback COCO
  'dog': 'Cachorro (raça não identificada)',
};

class DogScanApp extends StatelessWidget {
  const DogScanApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'DogScan',
      debugShowCheckedModeBanner: false,
      theme: ThemeData.dark(useMaterial3: true),
      home: const LivePage(),
    );
  }
}

class LivePage extends StatefulWidget {
  const LivePage({super.key});
  @override
  State<LivePage> createState() => _LivePageState();
}

class _LivePageState extends State<LivePage> {
  bool _granted = false;
  String _label = '';
  double _conf = 0;

  @override
  void initState() {
    super.initState();
    _askCamera();
  }

  Future<void> _askCamera() async {
    final status = await Permission.camera.request();
    setState(() => _granted = status.isGranted);
  }

  void _onResult(List<YOLOResult> results) {
    if (results.isEmpty) {
      if (_label.isNotEmpty) setState(() { _label = ''; _conf = 0; });
      return;
    }
    results.sort((a, b) => b.confidence.compareTo(a.confidence));
    final r = results.first;
    final pretty = kBreedPt[r.className] ?? r.className;
    setState(() { _label = pretty; _conf = r.confidence; });
  }

  @override
  Widget build(BuildContext context) {
    if (!_granted) {
      return Scaffold(
        body: Center(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const Text('Precisa de acesso à câmera'),
              const SizedBox(height: 12),
              FilledButton(onPressed: _askCamera, child: const Text('Permitir')),
            ],
          ),
        ),
      );
    }

    return Scaffold(
      body: Stack(
        children: [
          Positioned.fill(
            child: YOLOView(
              modelPath: kUseTrainedModel ? kTrainedModel : kCocoModel,
              task: YOLOTask.detect,
              confidenceThreshold: kConfidence,
              onResult: _onResult,
            ),
          ),
          // aviso topo
          const Positioned(
            top: 44, left: 12, right: 12,
            child: Text(
              'Estimativa automática — pode errar.',
              textAlign: TextAlign.center,
              style: TextStyle(color: Colors.white70, fontSize: 12),
            ),
          ),
          // resultado rodape
          if (_label.isNotEmpty)
            Positioned(
              left: 12, right: 12, bottom: 28,
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
                decoration: BoxDecoration(
                  color: Colors.black.withValues(alpha: 0.6),
                  borderRadius: BorderRadius.circular(14),
                ),
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(_label,
                        style: const TextStyle(
                            fontSize: 20, fontWeight: FontWeight.bold)),
                    const SizedBox(height: 6),
                    ClipRRect(
                      borderRadius: BorderRadius.circular(6),
                      child: LinearProgressIndicator(
                        value: _conf,
                        minHeight: 6,
                        backgroundColor: Colors.white24,
                      ),
                    ),
                    const SizedBox(height: 4),
                    Text('${(_conf * 100).toStringAsFixed(0)}% de confiança',
                        style: const TextStyle(fontSize: 12, color: Colors.white70)),
                  ],
                ),
              ),
            ),
        ],
      ),
    );
  }
}
