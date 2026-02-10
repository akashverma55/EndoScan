from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
import torch.nn as nn
from torchvision import models
from timm import create_model
import cv2
import numpy as np
from PIL import Image
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware, 
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load models
resnet = models.resnet50(weights=None)
resnet.fc = nn.Linear(resnet.fc.in_features, 4)
resnet.load_state_dict(torch.load("models/ResNet50_best.pth", map_location=device, weights_only=False))
resnet.to(device).eval()

convnext = create_model('convnext_base', pretrained=False, num_classes=4)
convnext.load_state_dict(torch.load("models/ConvNext_best.pth", map_location=device, weights_only=False))
convnext.to(device).eval()

swin = create_model('swin_base_patch4_window7_224', pretrained=False, num_classes=4)
swin.load_state_dict(torch.load("models/Swin_best.pth", map_location=device, weights_only=False))
swin.to(device).eval()

maxvit = create_model('maxvit_base_tf_224', pretrained=False, num_classes=4)
maxvit.load_state_dict(torch.load("models/MaxViT_best.pth", map_location=device, weights_only=False))
maxvit.to(device).eval()

CLASSES = ['EA', 'EH', 'EP', 'NE']

def preprocess_image(image_bytes):
    """Preprocess image for model input"""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image = np.array(image)
    image = cv2.resize(image, (224, 224))
    image = image.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image = (image - mean) / std
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read file
        contents = await file.read()
        image_tensor = preprocess_image(contents).to(device)

        # Inference
        with torch.no_grad():
            pred1 = torch.softmax(resnet(image_tensor), dim=1)
            pred2 = torch.softmax(convnext(image_tensor), dim=1)
            pred3 = torch.softmax(swin(image_tensor), dim=1)
            pred4 = torch.softmax(maxvit(image_tensor), dim=1)

            # Ensemble
            ensemble_pred = (pred1 + pred2 + pred3 + pred4) / 4

            probs = ensemble_pred[0].cpu().numpy()
            predicted_class = int(np.argmax(probs))
            confidence = float(probs[predicted_class])

            return {
                "success": True,
                "predicted_class": CLASSES[predicted_class],
                "confidence": confidence,
                "probabilities": {
                    CLASSES[i]: float(probs[i]) for i in range(len(CLASSES))
                }
            }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }

@app.get("/")
def root():
    return {
        "message": "Welcome to the EndoVision API",
        "status": "healthy",
        "device": str(device)
    }

@app.get("/health")
def health():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "device": str(device),
        "models": {
            "resnet50": "loaded",
            "convnext": "loaded",
            "swin": "loaded",
            "maxvit": "loaded"
        },
        "classes": CLASSES
    }