
from torchvision import transforms
import torch
import requests
from PIL import Image
from io import BytesIO
import matplotlib.pyplot as plt

inference_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def predict_image_from_url(
    image_url,
    model,
    device,
    fruit_lookup,
    freshness_lookup
):
    model.eval()

    # Download image
    response = requests.get(image_url)
    # img = Image.open(BytesIO(response.content)).convert("RGB")
    image_path = "/content/download.jpg"
    img = Image.open(image_path).convert("RGB")


    # Show image
    plt.imshow(img)
    plt.axis("off")
    plt.show()

    # Transform
    img_tensor = inference_transform(img).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        fruit_logits, freshness_logits = model(img_tensor)

        fruit_pred = torch.argmax(fruit_logits, dim=1).item()
        freshness_pred = torch.argmax(freshness_logits, dim=1).item()

    # Decode labels
    print(fruit_pred)
    print(freshness_pred)
    print(fruit_to_idx)
    print(freshness_to_idx)
    fruit_label = list(fruit_to_idx.keys())[list(fruit_to_idx.values()).index(fruit_pred)]
    freshness_label = list(freshness_to_idx.keys())[list(freshness_to_idx.values()).index(freshness_pred)]

    print("Prediction Results")
    print(f"Fruit: {fruit_label}")
    print(f"Freshness: {freshness_label}")


# you can use this sample image
image_url = "https://upload.wikimedia.org/wikipedia/commons/1/15/Red_Apple.jpg"

predict_image_from_url(
    image_url=image_url,
    model=model,
    device=device,
    fruit_lookup=fruit_lookup,
    freshness_lookup=freshness_lookup
)