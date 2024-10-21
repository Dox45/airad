from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import torchvision.models as models
import torch.nn as nn
from torchvision import transforms
import io
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


app = FastAPI()
model_checkpoint = "model/health1.pth"
pre_computed_embeds = "model/comp_embed.npy"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
sim_model = models.resnet50(pretrained=True)
sim_model.to(device)
sim_model.eval() 
# Define class mapping
class_names = {0: 'normal', 1: 'pneumonia'}


#compute the embeddings of the incoming image
def get_embedding(image_path): 
    with torch.no_grad():  
        embedding = sim_model(image_path)  
    return embedding.cpu().numpy().flatten()  


#load pre_computed embeddings to filter non_xray images
def load_embedding(embedding):
  embed = np.load(embedding)
  return embed


# Load the model at startup
def load_model():
    model = models.efficientnet_b0(pretrained=False)
    num_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(num_features, 2)  
    )
    model.to(device)
    if device == 'cuda':
      model.load_state_dict(torch.load(model_checkpoint))
    else:
      model.load_state_dict(torch.load(model_checkpoint, map_location=torch.device('cpu')))
    model.eval()  
    return model

model = load_model()  # Load the model once when the app starts

# Preprocessing function
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# Endpoint to handle image upload and return prediction
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read the uploaded file as an image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Preprocess the image
        image = preprocess(image).unsqueeze(0) 
        image = image.to(device) 
        embedding1 = get_embedding(image)
        embedding2 = load_embedding(pre_computed_embeds)
        similarity = cosine_similarity([embedding1],[embedding2])[0][0]
        if similarity >= 0.60:

          # Make prediction
          with torch.no_grad():
              output = model(image)
              _, predicted = torch.max(output, 1)
              predicted_class = predicted.item()

          # Return the prediction as JSON
          return JSONResponse(content={"class": class_names[predicted_class]})
        else:
          print('model did not recognize image as xray, if you feel that this is an error report performance')
    
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
        
