from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms
import io
import numpy as np
import gdown
import os
import cv2

file_id = '1CKkdQ5nKWkz3L-ZdgyrJ5SE-oiFwXnSJ'
gdrive_url = f"https://drive.google.com/uc?id={file_id}"
model_checkpoint = 'fine_tuned_health.pth'

sim_file_id = '1yYlT-RgMNCprPOJiur4YoajroiRdB64V'
sim_gdrive_url = f"https://drive.google.com/uc?id={sim_file_id}"
sim_model_checkpoint = 'disc_2_embed.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Download model checkpoint if it doesn't exist
if not os.path.exists(model_checkpoint):
    gdown.download(gdrive_url, model_checkpoint, quiet=False)
if not os.path.exists(sim_model_checkpoint):
    gdown.download(sim_gdrive_url, sim_model_checkpoint,quiet=False)
# Initialize FastAPI app
app = FastAPI()

# Load EfficientNet model for classification
def load_classification_model():
    model = models.efficientnet_b0(pretrained=False)
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(num_features, 2)  # Two classes: normal, pneumonia
    )
    sim_model = model
    model.to(device)
    sim_model.to(device)
    if device == 'cuda':
        model.load_state_dict(torch.load(model_checkpoint))
        sim_model.load_state_dict(torch.load(sim_model_checkpoint))
    else:
        model.load_state_dict(torch.load(model_checkpoint, map_location=torch.device('cpu')))
        sim_model.load_state_dict(torch.load(sim_model_checkpoint, map_location=torch.device('cpu')))
    model.eval()
    sim_model.eval()
    return model, sim_model

classification_model, similarity_model = load_classification_model()  # EfficientNet for classification

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the uploaded file as an image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_np = np.array(image)
        image_np = cv2.bilateralFilter(image_np, 9, 75, 75)
        image = Image.fromarray(image_np)

        # Preprocess the image
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            is_xray = similarity_model(image_tensor)
            _, predicted_ray = torch.max(is_xray,1)
            if predicted_ray.item() == 1:
                with torch.no_grad():
                    output = classification_model(image_tensor)
                    _, predicted = torch.max(output, 1)
                    predicted_class = predicted.item()
                    class_names = {0: 'normal', 1: 'pneumonia'}
                    confidence = torch.nn.functional.softmax(output, dim=1)[0][predicted_class].item()

                    # Return the prediction as JSON
                    return JSONResponse(
                        content={
                            "status": "success",
                            "message": "Prediction successful.",
                            "data": {
                                "class": class_names[predicted_class],
                                "confidence": confidence
                            }
                        },
                        status_code=200
                    )
            else:
                return JSONResponse(
                    content={
                        "status": "failed",
                        "message": "Image is likely not an xray image."
                    },
                status_code=400
            )

    except Exception as e:
        return JSONResponse(
            content={
                "status": "failed",
                "error": str(e),
                "message": "An error occurred during prediction."
            },
            status_code=500
        )
