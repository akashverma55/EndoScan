import 'package:endoapp/api_service.dart';
import 'package:endoapp/bloc/classification_bloc.dart';
import 'package:endoapp/classification_screen.dart';
import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Endometrial Classification',
      debugShowCheckedModeBanner: false,
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.deepPurple),
        useMaterial3: true,
      ),
      home: BlocProvider(
        create: (context) => ClassificationBloc(
          api: EndometrialClassificationAPI(
            baseUrl: 'http://192.168.1.5:8000', 
          ),
        ),
        child: const ClassificationScreen(),
      ),
    );
  }
}