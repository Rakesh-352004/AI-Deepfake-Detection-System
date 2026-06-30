import torch
import torchvision.transforms as transforms
from PIL import Image
import torchvision
import cv2

# Load model
model = torchvision.models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, 2)

model.load_state_dict(torch.load("deepfake_model.pth", map_location="cpu"))
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# Face detector
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

def detect_face(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return image

    x, y, w, h = faces[0]
    return image[y:y+h, x:x+w]

def predict(image):

    face = detect_face(image)

    image = Image.fromarray(face)
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        probs = torch.softmax(output, dim=1)

        confidence, pred = torch.max(probs, 1)

    label = "Real" if pred.item() == 0 else "Fake"
    confidence = round(confidence.item() * 100, 2)

    return label, confidence