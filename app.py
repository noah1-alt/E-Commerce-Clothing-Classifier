from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os
import logging

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limit uploads to 16MB
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Define the CNN class
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
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {e}")
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
    InvertImage(),
])

# Class labels
classes = ['T-shirt/Top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    error = None
    sample_images = []
    uploaded_image = None  # Ensure this is defined

    try:
        sample_images = [f for f in os.listdir('static/sample_images') if allowed_file(f)]
        logger.info(f"Sample images: {sample_images}")
    except Exception as e:
        logger.error(f"Error listing sample images: {e}")
        error = "Unable to load sample images"

    if request.method == 'POST':
        if 'image' not in request.files:
            logger.warning("No file received in POST request")
            error = "No file uploaded"
        else:
            file = request.files['image']
            if file.filename == '':
                logger.warning("Empty filename received")
                error = "No file selected"
            elif not allowed_file(file.filename):
                logger.warning(f"Invalid file extension: {file.filename}")
                error = "Invalid file format. Please upload PNG, JPG, or JPEG"
            else:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                try:
                    file.save(filepath)
                    logger.info(f"File saved: {filepath}")
                    img = Image.open(filepath)
                    logger.info(f"Image mode: {img.mode}")
                    img_tensor = transform(img).unsqueeze(0)
                    logger.info(f"Input tensor shape: {img_tensor.shape}")
                    logger.info(f"Input tensor min/max: {img_tensor.min().item()} {img_tensor.max().item()}")
                    try:
                        with torch.no_grad():
                            outputs = model(img_tensor)
                            logger.info(f"Output shape: {outputs.shape}")
                            logger.info(f"Output logits: {outputs.tolist()}")
                            probabilities = torch.softmax(outputs, dim=1)
                            logger.info(f"Output probabilities: {probabilities.tolist()}")
                            confidence, predicted = torch.max(probabilities, 1)
                            prediction = classes[predicted.item()]
                            confidence = confidence.item() * 100
                            logger.info(f"Predicted index: {predicted.item()}")
                            logger.info(f"Prediction: {prediction} ({confidence:.2f}%)")
                    except Exception as e:
                        logger.error(f"Error during prediction: {e}")
                        error = "Prediction failed"
                    finally:
                        try:
                            os.remove(filepath)  # Clean up uploaded file
                            logger.info(f"File deleted: {filepath}")
                        except Exception as e:
                            logger.error(f"Error deleting file: {e}")
                except Exception as e:
                    logger.error(f"Error processing image: {e}")
                    error = "Error processing image"
                uploaded_image = filename  # Set the filename to show the uploaded image

    return render_template('index.html', prediction=prediction, confidence=confidence, error=error, sample_images=sample_images, uploaded_image=uploaded_image)

@app.route('/predict_sample/<filename>')
def predict_sample(filename):
    prediction = None
    confidence = None
    error = None
    sample_images = []

    try:
        sample_images = [f for f in os.listdir('static/sample_images') if allowed_file(f)]
        logger.info(f"Sample images: {sample_images}")
    except Exception as e:
        logger.error(f"Error listing sample images: {e}")
        error = "Unable to load sample images"

    filepath = os.path.join('static/sample_images', filename)
    if not allowed_file(filename):
        logger.warning(f"Invalid file extension: {filename}")
        error = "Invalid sample image format"
    else:
        try:
            img = Image.open(filepath)
            logger.info(f"Sample image opened: {filepath}")
            logger.info(f"Image mode: {img.mode}")
            img_tensor = transform(img).unsqueeze(0)
            logger.info(f"Input tensor shape: {img_tensor.shape}")
            logger.info(f"Input tensor min/max: {img_tensor.min().item()} {img_tensor.max().item()}")
            try:
                with torch.no_grad():
                    outputs = model(img_tensor)
                    logger.info(f"Output shape: {outputs.shape}")
                    logger.info(f"Output logits: {outputs.tolist()}")
                    probabilities = torch.softmax(outputs, dim=1)
                    logger.info(f"Output probabilities: {probabilities.tolist()}")
                    confidence, predicted = torch.max(probabilities, 1)
                    prediction = classes[predicted.item()]
                    confidence = confidence.item() * 100
                    logger.info(f"Predicted index: {predicted.item()}")
                    logger.info(f"Prediction: {prediction} ({confidence:.2f}%)")
            except Exception as e:
                logger.error(f"Error during prediction: {e}")
                error = "Prediction failed"
        except Exception as e:
            logger.error(f"Error opening sample image: {e}")
            error = "Error loading sample image"

    return render_template('index.html', prediction=prediction, confidence=confidence, error=error, sample_images=sample_images)

if __name__ == '__main__':
    app.run(debug=False)
