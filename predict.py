import torch

from PIL import Image
from transformers import (TrOCRProcessor, VisionEncoderDecoderModel)

# SETTINGS
MODEL_PATH = "./etrocr_model"
IMAGE_PATH = "./test_image.png"


# DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

# LOAD PROCESSOR + MODEL
processor = TrOCRProcessor.from_pretrained(MODEL_PATH)
model = (VisionEncoderDecoderModel.from_pretrained(MODEL_PATH))
model.to(device)
model.eval()


# LOAD IMAGE
image = Image.open(IMAGE_PATH).convert("RGB")


# PREPROCESS IMAGE
pixel_values = processor(image, return_tensors="pt").pixel_values.to(device)


# GENERATE PREDICTION
with torch.no_grad():
    generated_ids = model.generate(pixel_values)

# DECODE TEXT
predicted_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

# OUTPUT
print("\nPredicted Text:")
print(predicted_text)