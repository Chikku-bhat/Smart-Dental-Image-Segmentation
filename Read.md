# Smart Dental Image Segmentation

## Introduction

This project aims to generate tooth masks from dental panoramic radiographs using deep learning techniques. The project consists of two main components: a backend for model training and a frontend web application for user interaction. The backend involves training a Generative Adversarial Network (GAN) model to generate tooth masks, while the frontend provides a user-friendly interface for uploading radiographs, generating masks, and visualizing the results.

## Dependencies and Installation

To run the backend, ensure the following dependencies are installed:
- TensorFlow (2.14.x)
- Flask
- NumPy
- Pillow (PIL)

To install dependencies, run:
pip install tensorflow flask numpy pillow

To run the frontend web application, additional dependencies are required:
- Flask

To install Flask, run:
pip install flask

Additionally, the trained model files (`final_tooth_mask_generation.h5` and `mask_generator.h5`) need to be available in the project directory.

## Usage

### Backend (Model Training)

1. Run the provided Python script to train the GAN model for tooth mask generation from dental panoramic radiographs.
2. Ensure the model is properly trained and saved before proceeding to the frontend.

### Frontend (Web Application)

1. Run the Flask web application by executing the provided Python script:
python app.py

2. Access the web application through the provided URL (usually http://localhost:5000) in a web browser.
3. Use the interface to upload dental panoramic radiographs.
4. Upon upload, the application generates tooth masks using the trained model and displays the results.
5. Users can visualize the original image, generated mask, and original mask (if available) for comparison.

## Methodology

The backend of the project involves training a Generative Adversarial Network (GAN) model to generate tooth masks from dental panoramic radiographs. The GAN consists of a generator and a discriminator, trained adversarially to improve mask generation accuracy. The frontend is implemented as a Flask web application, providing an intuitive interface for users to upload radiographs, generate masks, and visualize the results. The Flask app communicates with the backend to process uploaded images, generate masks using the trained model, and display the results to the user.
