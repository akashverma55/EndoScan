import 'dart:io';
import 'package:dio/dio.dart';
import 'package:endoapp/response_model.dart';

class EndometrialClassificationAPI {
  final Dio _dio;
  final String baseUrl;

  EndometrialClassificationAPI({required this.baseUrl}) : _dio = Dio(
          BaseOptions(
            baseUrl: baseUrl,
            connectTimeout: const Duration(seconds: 60),
            receiveTimeout: const Duration(seconds: 60),
            sendTimeout: const Duration(seconds: 60),
            headers: {
              'Accept': 'application/json',
            },
          ),
        );

  Future<ClassificationResponse> classifyImage(File imageFile) async {
    try {
      // Check if file exists
      if (!await imageFile.exists()) {
        return ClassificationResponse(
          success: false,
          error: 'Image file not found',
        );
      }

      // Create form data
      FormData formData = FormData.fromMap({
        'file': await MultipartFile.fromFile(
          imageFile.path,
          filename: imageFile.path.split('/').last,
        ),
      });

      // Send POST request
      Response response = await _dio.post(
        '/predict',
        data: formData,
        options: Options(
          headers: {
            'Content-Type': 'multipart/form-data',
          },
        ),
      );

      // Return response data
      return ClassificationResponse.fromJson(response.data);
      
    } on DioException catch (e) {
      // Handle Dio-specific errors
      return ClassificationResponse(
        success: false,
        error: _handleDioError(e)['error'],
      );
    } catch (e) {
      // Handle other errors
      return ClassificationResponse(
        success: false,
        error: 'Unexpected error: ${e.toString()}',
      );
    }
  }

  /// Handle Dio errors
  Map<String, dynamic> _handleDioError(DioException e) {
    switch (e.type) {
      case DioExceptionType.connectionTimeout:
        return {
          'success': false,
          'error': 'Connection timeout. Please check your network connection.',
        };
      case DioExceptionType.sendTimeout:
        return {
          'success': false,
          'error': 'Request timeout. Please try again.',
        };
      case DioExceptionType.receiveTimeout:
        return {
          'success': false,
          'error': 'Server response timeout. Please try again.',
        };
      case DioExceptionType.badResponse:
        return {
          'success': false,
          'error': 'Server error: ${e.response?.statusCode}',
          'details': e.response?.data,
        };
      case DioExceptionType.connectionError:
        return {
          'success': false,
          'error': 'Cannot connect to server. Please check your connection.',
        };
      case DioExceptionType.cancel:
        return {
          'success': false,
          'error': 'Request was cancelled',
        };
      default:
        return {
          'success': false,
          'error': 'Network error: ${e.message}',
        };
    }
  }

  /// Check server health
  Future<bool> checkServerHealth() async {
    try {
      final response = await _dio.get('/');
      return response.statusCode == 200;
    } catch (e) {
      return false;
    }
  }
}