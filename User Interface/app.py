from flask import Flask, render_template, request, send_from_directory
from PIL import Image
import numpy as np
import os
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the model
model = load_model("final_tooth_mask_generation.h5")
model.load_weights("mask_generator.h5")

# Define folder paths
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'outputs'
ORIGINAL_MASK_FOLDER = 'original_mask'  # Add original mask folder
NEW_ORIGINAL_MASK_FOLDER='new_ori_mask'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
if not os.path.exists(ORIGINAL_MASK_FOLDER):
    os.makedirs(ORIGINAL_MASK_FOLDER)
if not os.path.exists(NEW_ORIGINAL_MASK_FOLDER):
    os.makedirs(NEW_ORIGINAL_MASK_FOLDER)

def preprocess_input(image_path):
    img = Image.open(image_path)
    img = img.convert('L')
    img = img.resize((224, 224))
    img = np.array(img)
    img = img.astype('float32') / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    return img

def resize_image(image_path):
    img = Image.open(image_path)
    img = img.resize((224, 224))
    return img

def blended_image(uploaded_image, masked_image):
    img1=uploaded_image.convert('L')
    img2=masked_image.convert('L')
    alpha=0.6
    blended_image = Image.blend(img1, img2, alpha)
    return blended_image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = file.filename
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)

            # Resize uploaded image
            img = resize_image(file_path)
            img.save(file_path)

            img_processed = preprocess_input(file_path)

            masked_img = model.predict(img_processed)

            masked_img = (masked_img[0] * 255).astype(np.uint8)
            masked_img = np.squeeze(masked_img)
            masked_img=Image.fromarray(masked_img)

            blend1=blended_image(img,masked_img)

            output_file_path = os.path.join(OUTPUT_FOLDER, filename)
            blend1.save(output_file_path)

            # Check if corresponding image exists in original mask folder
            original_mask_filename = None
            if os.path.exists(os.path.join(ORIGINAL_MASK_FOLDER, filename)):
               original_mask_filename = filename
               original_mask_img = resize_image(os.path.join(ORIGINAL_MASK_FOLDER, filename))
               blend2=blended_image(img,original_mask_img)
               blend2.save(os.path.join(NEW_ORIGINAL_MASK_FOLDER, filename))

        return render_template('result.html', original_image=filename, masked_image=filename, original_mask_image=original_mask_filename)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route('/outputs/<filename>')
def output_file(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

@app.route('/new_ori_mask/<filename>')
def original_mask_file(filename):
    return send_from_directory(NEW_ORIGINAL_MASK_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True)


