import cv2
import numpy as np
import streamlit as st
from PIL import Image

def detect_blurriness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var

def detect_brightness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    return brightness

def detect_contrast(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    contrast = gray.std()
    return contrast

def detect_noise(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    noise = cv2.Laplacian(gray, cv2.CV_64F).var()
    return noise

def detect_skew(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    
    angles = []
    if lines is not None:
        for rho, theta in lines[:, 0]:
            angle = np.degrees(theta) - 90
            angles.append(angle)
    
    return np.median(angles) if angles else 0  # Median angle as final skew value

def detect_curve(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    y_points = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        y_points.append(y + h // 2)  # Midpoint of text
    
    if len(y_points) > 2:
        poly_coeffs = np.polyfit(range(len(y_points)), y_points, deg=2)  # Quadratic fit
        curvature = abs(poly_coeffs[0])  # Curvature coefficient
    else:
        curvature = 0  # No curve detected
    
    return curvature  # Higher values mean more curvature

st.title("Image Quality Checker")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_np = np.array(image)
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    blurriness = detect_blurriness(image_cv)
    brightness = detect_brightness(image_cv)
    contrast = detect_contrast(image_cv)
    noise = detect_noise(image_cv)
    skew_angle = detect_skew(image_cv)
    curvature = detect_curve(image_cv)
    
    st.subheader("Image Quality Analysis")
    st.write(f"**Blurriness:** {'Sharp' if blurriness > 100 else 'Blurry'} (Score: {blurriness:.2f}, Recommended: >100)")
    st.write(f"**Brightness:** {'Good' if 50 <= brightness <= 200 else 'Too Dark' if brightness < 50 else 'Too Bright'} (Value: {brightness:.2f}, Recommended: 100-200)")
    st.write(f"**Contrast:** {'High' if contrast > 50 else 'Low'} (Value: {contrast:.2f}, Recommended: >50)")
    st.write(f"**Noise Level:** {'High' if noise > 150 else 'Low'} (Score: {noise:.2f}, Recommended: <150)")
    st.write(f"**Skew Angle:** {'Aligned' if abs(skew_angle) < 5 else 'Misaligned'} (Angle: {skew_angle:.2f}°, Recommended: ±5°)")
    st.write(f"**Text Curvature:** {'Straight' if curvature < 0.01 else 'Curved'} (Curvature: {curvature:.4f}, Recommended: <0.01)")
    
    st.subheader("Uploaded Image")
    st.image(image, caption="Uploaded Image", use_column_width=True)