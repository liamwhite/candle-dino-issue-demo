from transformers import AutoModel, AutoImageProcessor
from PIL import Image
import time

MODEL_NAME = "facebook/dinov2-base"
DEVICE = "cpu"

if __name__ == "__main__":
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(DEVICE)

    image = Image.open("../image.png")
    inputs = processor(image, return_tensors="pt").to(DEVICE)

    begin = time.time()
    outputs = model(**inputs)
    end = time.time()

    print(f"Took {end - begin} seconds to evaluate")
