import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'dart:io';
import 'package:image_picker/image_picker.dart';
import 'package:endoapp/bloc/classification_bloc.dart';
import 'package:endoapp/bloc/classification_event.dart';
import 'package:endoapp/bloc/classification_state.dart';

class ClassificationScreen extends StatefulWidget {
  const ClassificationScreen({super.key});

  @override
  ClassificationScreenState createState() => ClassificationScreenState();
}

class ClassificationScreenState extends State<ClassificationScreen>
    with SingleTickerProviderStateMixin {
  File? _selectedImage;
  final ImagePicker _picker = ImagePicker();
  late AnimationController _animationController;
  late Animation<double> _fadeAnimation;

  @override
  void initState() {
    super.initState();
    _animationController = AnimationController(
      duration: const Duration(milliseconds: 600),
      vsync: this,
    );
    _fadeAnimation = CurvedAnimation(
      parent: _animationController,
      curve: Curves.easeInOut,
    );
  }

  @override
  void dispose() {
    _animationController.dispose();
    super.dispose();
  }

  /// Pick image from gallery
  Future<void> _pickImage() async {
    final XFile? image = await _picker.pickImage(
      source: ImageSource.gallery,
      imageQuality: 85,
    );

    if (image != null) {
      setState(() {
        _selectedImage = File(image.path);
      });
      _animationController.forward(from: 0);
      // Reset classification state when new image is selected
      context.read<ClassificationBloc>().add(ResetClassificationEvent());
    }
  }

  /// Take photo with camera
  Future<void> _takePhoto() async {
    final XFile? image = await _picker.pickImage(
      source: ImageSource.camera,
      imageQuality: 85,
    );

    if (image != null) {
      setState(() {
        _selectedImage = File(image.path);
      });
      _animationController.forward(from: 0);
      context.read<ClassificationBloc>().add(ResetClassificationEvent());
    }
  }

  /// Classify the selected image using BLoC
  void _classifyImage() {
    if (_selectedImage == null) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: const Row(
            children: [
              Icon(Icons.warning_amber_rounded, color: Colors.white),
              SizedBox(width: 12),
              Expanded(child: Text('Please select an image first')),
            ],
          ),
          backgroundColor: Colors.orange.shade700,
          behavior: SnackBarBehavior.floating,
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
          margin: const EdgeInsets.all(16),
        ),
      );
      return;
    }

    // Trigger classification event
    context.read<ClassificationBloc>().add(ClassifyImageEvent(_selectedImage!));
  }

  void _showImageSourceDialog() {
    showModalBottomSheet(
      context: context,
      backgroundColor: Colors.transparent,
      builder: (context) => Container(
        decoration: BoxDecoration(
          color: const Color(0xFF1E293B),
          borderRadius: const BorderRadius.vertical(top: Radius.circular(24)),
          border: Border.all(color: Colors.cyanAccent.withOpacity(0.2)),
        ),
        child: SafeArea(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              const SizedBox(height: 12),
              Container(
                width: 40,
                height: 4,
                decoration: BoxDecoration(
                  color: Colors.cyanAccent.withOpacity(0.3),
                  borderRadius: BorderRadius.circular(2),
                ),
              ),
              const SizedBox(height: 20),
              const Text(
                'Select Image Source',
                style: TextStyle(
                  fontSize: 20,
                  fontWeight: FontWeight.bold,
                  color: Colors.white,
                ),
              ),
              const SizedBox(height: 20),
              ListTile(
                leading: Container(
                  padding: const EdgeInsets.all(10),
                  decoration: BoxDecoration(
                    color: Colors.cyanAccent.withOpacity(0.1),
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(
                      color: Colors.cyanAccent.withOpacity(0.3),
                    ),
                  ),
                  child: const Icon(Icons.photo_library, color: Colors.cyanAccent),
                ),
                title: const Text(
                  'Gallery',
                  style: TextStyle(color: Colors.white, fontWeight: FontWeight.w600),
                ),
                subtitle: const Text(
                  'Choose from your photos',
                  style: TextStyle(color: Colors.white60),
                ),
                onTap: () {
                  Navigator.pop(context);
                  _pickImage();
                },
              ),
              ListTile(
                leading: Container(
                  padding: const EdgeInsets.all(10),
                  decoration: BoxDecoration(
                    color: Colors.purpleAccent.withOpacity(0.1),
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(
                      color: Colors.purpleAccent.withOpacity(0.3),
                    ),
                  ),
                  child: const Icon(Icons.camera_alt, color: Colors.purpleAccent),
                ),
                title: const Text(
                  'Camera',
                  style: TextStyle(color: Colors.white, fontWeight: FontWeight.w600),
                ),
                subtitle: const Text(
                  'Take a new photo',
                  style: TextStyle(color: Colors.white60),
                ),
                onTap: () {
                  Navigator.pop(context);
                  _takePhoto();
                },
              ),
              const SizedBox(height: 20),
            ],
          ),
        ),
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Theme(
      data: ThemeData.dark().copyWith(
        scaffoldBackgroundColor: const Color(0xFF0F172A), // Modern Deep Navy
        cardColor: const Color(0xFF1E293B),
      ),
      child: Scaffold(
        appBar: AppBar(
          elevation: 0,
          backgroundColor: Colors.transparent,
          title: const Text(
            'AI ANALYSIS',
            style: TextStyle(
              letterSpacing: 1.2,
              fontWeight: FontWeight.bold,
            ),
          ),
          centerTitle: true,
        ),
        body: SingleChildScrollView(
          physics: const BouncingScrollPhysics(),
          child: Column(
            children: [
              _buildModernHeader(),
              Padding(
                padding: const EdgeInsets.symmetric(horizontal: 20),
                child: _buildImagePreview(),
              ),
              const SizedBox(height: 24),
              _buildActionRow(context),
              const SizedBox(height: 24),
              _buildResultsSection(),
            ],
          ),
        ),
      ),
    );
  }

  // Modern Glassmorphism Header
  Widget _buildModernHeader() {
    return Container(
      margin: const EdgeInsets.all(20),
      padding: const EdgeInsets.all(24),
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(24),
        gradient: LinearGradient(
          colors: [
            Colors.cyan.withOpacity(0.15),
            Colors.blueAccent.withOpacity(0.05)
          ],
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
        ),
        border: Border.all(color: Colors.cyan.withOpacity(0.2)),
      ),
      child: Column(
        children: [
          const Icon(Icons.auto_awesome, color: Colors.cyanAccent, size: 40),
          const SizedBox(height: 12),
          Text(
            'ENDOMETRIAL SCAN',
            style: TextStyle(
              color: Colors.cyanAccent.shade100,
              fontWeight: FontWeight.w900,
              letterSpacing: 2,
              fontSize: 12,
            ),
          ),
          const SizedBox(height: 8),
          const Text(
            'Neural Classification Engine',
            style: TextStyle(
              color: Colors.white,
              fontSize: 18,
              fontWeight: FontWeight.w300,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildImagePreview() {
    if (_selectedImage != null) {
      return FadeTransition(
        opacity: _fadeAnimation,
        child: Hero(
          tag: 'selected_image',
          child: Container(
            height: 320,
            decoration: BoxDecoration(
              borderRadius: BorderRadius.circular(24),
              border: Border.all(color: Colors.cyanAccent.withOpacity(0.3), width: 1.5),
              boxShadow: [
                BoxShadow(
                  color: Colors.cyanAccent.withOpacity(0.1),
                  blurRadius: 20,
                  spreadRadius: 2,
                  offset: const Offset(0, 10),
                ),
              ],
            ),
            child: ClipRRect(
              borderRadius: BorderRadius.circular(24),
              child: Stack(
                fit: StackFit.expand,
                children: [
                  Image.file(
                    _selectedImage!,
                    fit: BoxFit.cover,
                  ),
                  // Gradient overlay for better visibility
                  Container(
                    decoration: BoxDecoration(
                      gradient: LinearGradient(
                        begin: Alignment.topCenter,
                        end: Alignment.bottomCenter,
                        colors: [
                          Colors.black.withOpacity(0.3),
                          Colors.transparent,
                          Colors.black.withOpacity(0.5),
                        ],
                      ),
                    ),
                  ),
                  Positioned(
                    top: 12,
                    right: 12,
                    child: Container(
                      decoration: BoxDecoration(
                        color: const Color(0xFF1E293B).withOpacity(0.8),
                        borderRadius: BorderRadius.circular(20),
                        border: Border.all(
                          color: Colors.cyanAccent.withOpacity(0.3),
                        ),
                      ),
                      child: IconButton(
                        icon: const Icon(Icons.close, color: Colors.cyanAccent),
                        onPressed: () {
                          setState(() {
                            _selectedImage = null;
                          });
                          context
                              .read<ClassificationBloc>()
                              .add(ResetClassificationEvent());
                        },
                      ),
                    ),
                  ),
                ],
              ),
            ),
          ),
        ),
      );
    } else {
      return Container(
        height: 320,
        decoration: BoxDecoration(
          color: const Color(0xFF1E293B),
          borderRadius: BorderRadius.circular(24),
          border: Border.all(color: Colors.cyan.withOpacity(0.2), width: 2),
          boxShadow: [
            BoxShadow(
              color: Colors.cyan.withOpacity(0.05),
              blurRadius: 10,
              offset: const Offset(0, 5),
            ),
          ],
        ),
        child: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Container(
                padding: const EdgeInsets.all(24),
                decoration: BoxDecoration(
                  color: Colors.cyanAccent.withOpacity(0.1),
                  shape: BoxShape.circle,
                  border: Border.all(
                    color: Colors.cyanAccent.withOpacity(0.3),
                    width: 2,
                  ),
                ),
                child: Icon(
                  Icons.image_outlined,
                  size: 64,
                  color: Colors.cyanAccent.withOpacity(0.6),
                ),
              ),
              const SizedBox(height: 20),
              const Text(
                'No Image Selected',
                style: TextStyle(
                  fontSize: 18,
                  fontWeight: FontWeight.w600,
                  color: Colors.white70,
                ),
              ),
              const SizedBox(height: 8),
              Text(
                'Tap the button below to select',
                style: TextStyle(
                  fontSize: 14,
                  color: Colors.white.withOpacity(0.4),
                ),
              ),
            ],
          ),
        ),
      );
    }
  }

  Widget _buildActionRow(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 20),
      child: Row(
        children: [
          Expanded(
            child: _buildActionButton(
              icon: Icons.add_photo_alternate,
              label: 'Select Image',
              color: Colors.cyanAccent,
              onPressed: _showImageSourceDialog,
            ),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: BlocBuilder<ClassificationBloc, ClassificationState>(
              builder: (context, state) {
                final isLoading = state is ClassificationLoading;
                return _buildActionButton(
                  icon: Icons.analytics,
                  label: isLoading ? 'Analyzing...' : 'Analyze',
                  color: Colors.purpleAccent,
                  onPressed: isLoading ? null : _classifyImage,
                  isLoading: isLoading,
                );
              },
            ),
          ),
        ],
      ),
    );
  }

  // Neon-Style Action Button
  Widget _buildActionButton({
    required IconData icon,
    required String label,
    required Color color,
    required VoidCallback? onPressed,
    bool isLoading = false,
  }) {
    final bool isDisabled = onPressed == null;
    return AnimatedContainer(
      duration: const Duration(milliseconds: 200),
      height: 58,
      decoration: BoxDecoration(
        borderRadius: BorderRadius.circular(18),
        color: isDisabled ? Colors.white10 : color.withOpacity(0.1),
        border: Border.all(
          color: isDisabled ? Colors.transparent : color.withOpacity(0.5),
          width: 1.5,
        ),
        boxShadow: isDisabled
            ? []
            : [
                BoxShadow(
                  color: color.withOpacity(0.15),
                  blurRadius: 15,
                  spreadRadius: 1,
                )
              ],
      ),
      child: Material(
        color: Colors.transparent,
        child: InkWell(
          onTap: onPressed,
          borderRadius: BorderRadius.circular(18),
          child: Center(
            child: isLoading
                ? const SizedBox(
                    width: 20,
                    height: 20,
                    child: CircularProgressIndicator(
                      strokeWidth: 2,
                      color: Colors.white,
                    ),
                  )
                : Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Icon(
                        icon,
                        color: isDisabled ? Colors.white38 : color,
                        size: 20,
                      ),
                      const SizedBox(width: 10),
                      Text(
                        label,
                        style: TextStyle(
                          color: isDisabled ? Colors.white38 : Colors.white,
                          fontWeight: FontWeight.bold,
                          fontSize: 15,
                        ),
                      ),
                    ],
                  ),
          ),
        ),
      ),
    );
  }

  Widget _buildResultsSection() {
    return BlocConsumer<ClassificationBloc, ClassificationState>(
      listener: (context, state) {
        if (state is ClassificationError) {
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Row(
                children: [
                  const Icon(Icons.error_outline, color: Colors.white),
                  const SizedBox(width: 12),
                  Expanded(child: Text(state.message)),
                ],
              ),
              backgroundColor: Colors.red.shade600,
              behavior: SnackBarBehavior.floating,
              shape: RoundedRectangleBorder(
                borderRadius: BorderRadius.circular(12),
              ),
              margin: const EdgeInsets.all(16),
              duration: const Duration(seconds: 4),
            ),
          );
        }
      },
      builder: (context, state) {
        if (state is ClassificationSuccess) {
          return Padding(
            padding: const EdgeInsets.fromLTRB(20, 0, 20, 24),
            child: _buildResultCard(state),
          );
        }
        return const SizedBox.shrink();
      },
    );
  }

  Widget _buildResultCard(ClassificationSuccess state) {
    final result = state.response;

    return TweenAnimationBuilder<double>(
      tween: Tween(begin: 0.0, end: 1.0),
      duration: const Duration(milliseconds: 600),
      curve: Curves.easeOutCubic,
      builder: (context, value, child) {
        return Transform.scale(
          scale: value,
          child: Opacity(
            opacity: value,
            child: child,
          ),
        );
      },
      child: Container(
        decoration: BoxDecoration(
          color: const Color(0xFF1E293B),
          borderRadius: BorderRadius.circular(24),
          border: Border.all(
            color: Colors.greenAccent.withOpacity(0.3),
            width: 1.5,
          ),
          boxShadow: [
            BoxShadow(
              color: Colors.greenAccent.withOpacity(0.1),
              blurRadius: 20,
              spreadRadius: 2,
              offset: const Offset(0, 8),
            ),
          ],
        ),
        child: Column(
          children: [
            // Header with Glassmorphism
            Container(
              padding: const EdgeInsets.all(20),
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  colors: [
                    Colors.greenAccent.withOpacity(0.2),
                    Colors.green.withOpacity(0.1)
                  ],
                  begin: Alignment.topLeft,
                  end: Alignment.bottomRight,
                ),
                borderRadius:
                    const BorderRadius.vertical(top: Radius.circular(24)),
                border: Border(
                  bottom: BorderSide(
                    color: Colors.greenAccent.withOpacity(0.2),
                  ),
                ),
              ),
              child: Row(
                children: [
                  Container(
                    padding: const EdgeInsets.all(12),
                    decoration: BoxDecoration(
                      color: Colors.greenAccent.withOpacity(0.1),
                      shape: BoxShape.circle,
                      border: Border.all(
                        color: Colors.greenAccent.withOpacity(0.3),
                        width: 2,
                      ),
                    ),
                    child: const Icon(
                      Icons.check_circle,
                      color: Colors.greenAccent,
                      size: 32,
                    ),
                  ),
                  const SizedBox(width: 16),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          'CLASSIFICATION RESULT',
                          style: TextStyle(
                            color: Colors.greenAccent.withOpacity(0.7),
                            fontSize: 11,
                            fontWeight: FontWeight.w900,
                            letterSpacing: 1.5,
                          ),
                        ),
                        const SizedBox(height: 4),
                        Text(
                          result.predictedClass,
                          style: const TextStyle(
                            color: Colors.white,
                            fontSize: 22,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                      ],
                    ),
                  ),
                ],
              ),
            ),

            // Confidence Score
            Padding(
              padding: const EdgeInsets.all(20),
              child: Column(
                children: [
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      const Text(
                        'Confidence Level',
                        style: TextStyle(
                          fontSize: 16,
                          fontWeight: FontWeight.w600,
                          color: Colors.white70,
                        ),
                      ),
                      Text(
                        '${(result.confidence * 100).toStringAsFixed(1)}%',
                        style: const TextStyle(
                          fontSize: 24,
                          fontWeight: FontWeight.bold,
                          color: Colors.greenAccent,
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 12),
                  ClipRRect(
                    borderRadius: BorderRadius.circular(10),
                    child: TweenAnimationBuilder<double>(
                      tween: Tween(begin: 0.0, end: result.confidence),
                      duration: const Duration(milliseconds: 1200),
                      curve: Curves.easeOutCubic,
                      builder: (context, value, child) {
                        return Container(
                          height: 12,
                          decoration: BoxDecoration(
                            color: Colors.white.withOpacity(0.05),
                            borderRadius: BorderRadius.circular(10),
                          ),
                          child: Stack(
                            children: [
                              FractionallySizedBox(
                                widthFactor: value,
                                child: Container(
                                  decoration: BoxDecoration(
                                    gradient: LinearGradient(
                                      colors: [
                                        Colors.greenAccent,
                                        Colors.green.shade300,
                                      ],
                                    ),
                                    borderRadius: BorderRadius.circular(10),
                                    boxShadow: [
                                      BoxShadow(
                                        color: Colors.greenAccent.withOpacity(0.5),
                                        blurRadius: 8,
                                        spreadRadius: 1,
                                      ),
                                    ],
                                  ),
                                ),
                              ),
                            ],
                          ),
                        );
                      },
                    ),
                  ),
                ],
              ),
            ),

            Divider(
              height: 1,
              color: Colors.white.withOpacity(0.1),
            ),

            // All Probabilities
            Padding(
              padding: const EdgeInsets.all(20),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text(
                    'Detailed Probabilities',
                    style: TextStyle(
                      fontSize: 16,
                      fontWeight: FontWeight.bold,
                      color: Colors.white,
                    ),
                  ),
                  const SizedBox(height: 16),
                  ...result.probabilities.entries.map((entry) {
                    final percentage = entry.value * 100;
                    final isMax = entry.key == result.predictedClass;

                    return TweenAnimationBuilder<double>(
                      tween: Tween(begin: 0.0, end: entry.value),
                      duration: const Duration(milliseconds: 1000),
                      curve: Curves.easeOutCubic,
                      builder: (context, value, child) {
                        return Container(
                          margin: const EdgeInsets.only(bottom: 16),
                          padding: const EdgeInsets.all(16),
                          decoration: BoxDecoration(
                            color: isMax
                                ? Colors.greenAccent.withOpacity(0.08)
                                : Colors.white.withOpacity(0.03),
                            borderRadius: BorderRadius.circular(16),
                            border: Border.all(
                              color: isMax
                                  ? Colors.greenAccent.withOpacity(0.3)
                                  : Colors.white.withOpacity(0.1),
                              width: isMax ? 1.5 : 1,
                            ),
                          ),
                          child: Column(
                            children: [
                              Row(
                                mainAxisAlignment:
                                    MainAxisAlignment.spaceBetween,
                                children: [
                                  Row(
                                    children: [
                                      if (isMax)
                                        const Icon(
                                          Icons.star,
                                          color: Colors.greenAccent,
                                          size: 20,
                                        ),
                                      if (isMax) const SizedBox(width: 8),
                                      Text(
                                        entry.key,
                                        style: TextStyle(
                                          fontSize: 15,
                                          fontWeight: isMax
                                              ? FontWeight.bold
                                              : FontWeight.w500,
                                          color: isMax
                                              ? Colors.greenAccent
                                              : Colors.white70,
                                        ),
                                      ),
                                    ],
                                  ),
                                  Text(
                                    '${percentage.toStringAsFixed(1)}%',
                                    style: TextStyle(
                                      fontSize: 16,
                                      fontWeight: FontWeight.bold,
                                      color: isMax
                                          ? Colors.greenAccent
                                          : Colors.white70,
                                    ),
                                  ),
                                ],
                              ),
                              const SizedBox(height: 10),
                              ClipRRect(
                                borderRadius: BorderRadius.circular(6),
                                child: Container(
                                  height: 8,
                                  decoration: BoxDecoration(
                                    color: Colors.white.withOpacity(0.05),
                                    borderRadius: BorderRadius.circular(6),
                                  ),
                                  child: Stack(
                                    children: [
                                      FractionallySizedBox(
                                        widthFactor: value,
                                        child: Container(
                                          decoration: BoxDecoration(
                                            gradient: LinearGradient(
                                              colors: isMax
                                                  ? [
                                                      Colors.greenAccent,
                                                      Colors.green.shade300,
                                                    ]
                                                  : [
                                                      Colors.cyanAccent,
                                                      Colors.blue.shade300,
                                                    ],
                                            ),
                                            borderRadius: BorderRadius.circular(6),
                                            boxShadow: [
                                              BoxShadow(
                                                color: (isMax
                                                        ? Colors.greenAccent
                                                        : Colors.cyanAccent)
                                                    .withOpacity(0.3),
                                                blurRadius: 6,
                                              ),
                                            ],
                                          ),
                                        ),
                                      ),
                                    ],
                                  ),
                                ),
                              ),
                            ],
                          ),
                        );
                      },
                    );
                  }),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}