import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:dogscan/main.dart';

void main() {
  testWidgets('app sobe sem crashar', (tester) async {
    await tester.pumpWidget(const DogScanApp());
    expect(find.byType(MaterialApp), findsOneWidget);
  });
}
