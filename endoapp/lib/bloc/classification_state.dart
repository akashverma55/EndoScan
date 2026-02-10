import 'package:endoapp/response_model.dart';
import 'package:equatable/equatable.dart';

abstract class ClassificationState extends Equatable {
  const ClassificationState();
  
  @override
  List<Object?> get props => [];
}

class ClassificationInitial extends ClassificationState {}

class ClassificationLoading extends ClassificationState {}

class ClassificationSuccess extends ClassificationState {
  final ClassificationResponse response;
  const ClassificationSuccess(
    this.response,
  );

  @override
  List<Object> get props => [response];
}

class ClassificationError extends ClassificationState {
  final String message;

  const ClassificationError(this.message);

  @override
  List<Object> get props => [message];
}