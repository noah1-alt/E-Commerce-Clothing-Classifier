from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Define the CNN class (unchanged)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1) 
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 10)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.elu = nn.ELU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.pool(self.elu(self.conv1(x)))
        x = self.pool(self.elu(self.conv2(x)))
        x = self.pool(self.elu(self.bn3(self.conv3(x))))
        x = x.view(-1, 128 * 3 * 3)
        x = self.elu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Load the model
try:
    model = CNN()
    model.load_state_dict(torch.load('model/fashion_mnist_cnn.pth', map_location=torch.device('cpu')))
    model.eval()
    print("Model loaded successfully")
except Exception as e:
    print("Error loading model:", e)
    raise

# Custom transform to invert image
class InvertImage:
    def __call__(self, img):
        return 1.0 - img  # Invert: 1 (white) -> 0 (black), 0 (black) -> 1 (white)

# Define image transform with contrast enhancement
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-8)),  # Normalize contrast
    InvertImage(),  # Keep inversion for white-background images
])

# Class labels
classes = ['T-shirt/Top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    try:
        sample_images = os.listdir('static/sample_images')
        print("Sample images:", sample_images)
    except Exception as e:
        print("Error listing sample images:", e)
        sample_images = []

    if request.method == 'POST':
        if 'image' in request.files:
            file = request.files['image']
            print("File received:", file.filename)
            if file:
                filepath = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
                try:
                    file.save(filepath)
                    print("File saved:", filepath)
                    img = Image.open(filepath)
                    print("Image mode:", img.mode)
                    img_tensor = transform(img).unsqueeze(0)
                    print("Input tensor shape:", img_tensor.shape)
                    print("Input tensor min/max:", img_tensor.min().item(), img_tensor.max().item())
                    try:
                        with torch.no_grad():
                            outputs = model(img_tensor)
                            print("Output shape:", outputs.shape)
                            print("Output logits:", outputs.tolist())
                            probabilities = torch.softmax(outputs, dim=1)
                            print("Output probabilities:", probabilities.tolist())
                            _, predicted = torch.max(outputs, 1)
                            print("Predicted index:", predicted.item())
                            prediction = classes[predicted.item()]
                            print("Prediction:", prediction)
                    except Exception as e:
                        print("Error during prediction:", e)
                        prediction = "Prediction failed"
                except Exception as e:
                    print("Error processing image:", e)
                    prediction = "Error processing image"
        else:
            print("No file received in POST request")
            prediction = "No file uploaded"

    return render_template('index.html', prediction=prediction, sample_images=sample_images)

@app.route('/predict_sample/<filename>')
def predict_sample(filename):
    prediction = None
    try:
        sample_images = os.listdir('static/sample_images')
        print("Sample images:", sample_images)
    except Exception as e:
        print("Error listing sample images:", e)
        sample_images = []

    filepath = os.path.join('static/sample_images', filename)
    try:
        print("Attempting to open sample image:", filepath)
        img = Image.open(filepath)
        print("Sample image opened:", filepath)
        print("Image mode:", img.mode)
        img_tensor = transform(img).unsqueeze(0)
        print("Input tensor shape:", img_tensor.shape)
        print("Input tensor min/max:", img_tensor.min().item(), img_tensor.max().item())
        try:
            with torch.no_grad():
                outputs = model(img_tensor)
                print("Output shape:", outputs.shape)
                print("Output logits:", outputs.tolist())
                probabilities = torch.softmax(outputs, dim=1)
                print("Output probabilities:", probabilities.tolist())
                _, predicted = torch.max(outputs, 1)
                print("Predicted index:", predicted.item())
                prediction = classes[predicted.item()]
                print("Prediction:", prediction)
        except Exception as e:
            print("Error during prediction:", e)
            prediction = "Prediction failed"
    except Exception as e:
        print("Error opening sample image:", e)
        prediction = "Error loading sample image"

    return render_template('index.html', prediction=prediction, sample_images=sample_images)

if __name__ == '__main__':
    app.run(debug=True)