class ClassificationResponse{
  final bool success;
  final String predictedClass;
  final double confidence;
  final Map<String, double> probabilities;
  final String? error;

  ClassificationResponse({
    required this.success,
    this.predictedClass = '',
    this.confidence = 0.0,
    this.probabilities = const {},
    this.error,
  });

  factory ClassificationResponse.fromJson(Map<String, dynamic> json){
    if (json['success'] == false){
      return ClassificationResponse(
        success: false,
        error: json['error'] ?? 'Unknown API Error'
      );
    }
    return ClassificationResponse(
      success: json['success'],
      predictedClass: json['predicted_class'] ?? 'Unknown',
      confidence: (json['confidence'] as num?)?.toDouble() ?? 0.0,
      probabilities: (json['probabilities'] as Map<String, dynamic>).map(
        (key,value) => MapEntry(key, (value as num).toDouble())
      )
    );
  }
}