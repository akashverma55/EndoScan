import 'package:endoapp/api_service.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'classification_event.dart';
import 'classification_state.dart';

class ClassificationBloc extends Bloc<ClassificationEvent, ClassificationState> {
  final EndometrialClassificationAPI api;

  ClassificationBloc({required this.api}) : super(ClassificationInitial()) {
    on<ClassifyImageEvent>(_onClassifyImage);
    on<ResetClassificationEvent>(_onResetClassification);
  }

  Future<void> _onClassifyImage(
    ClassifyImageEvent event,
    Emitter<ClassificationState> emit,
  ) async {
    // Emit loading state
    emit(ClassificationLoading());

    try {
      // Call API to classify image
      final result = await api.classifyImage(event.imageFile);

      // Check if API call was successful
      if (result.success) {
        emit(ClassificationSuccess(result));
      } else {
        emit(ClassificationError(result.error ?? 'Unknown API Error'));
      }
    } catch (e) {
      // Handle any unexpected errors
      emit(ClassificationError('System Error: ${e.toString()}'));
    }
  }

  void _onResetClassification(
    ResetClassificationEvent event,
    Emitter<ClassificationState> emit,
  ) {
    // Reset to initial state
    emit(ClassificationInitial());
  }
}