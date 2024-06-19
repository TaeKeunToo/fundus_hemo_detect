pip install streamlit opencv-python-headless numpy scikit-image pyngrok


%%writefile app.py
import streamlit as st
import cv2
import numpy as np
from skimage.feature import corner_harris, corner_peaks
from skimage.filters import gabor

def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    processed_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return processed_image

def detect_features(image, radii_range, sensitivity=0.95, edge_threshold=0.1):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    # Circular Hough Transform
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=15,
        param1=30,
        param2=15,
        minRadius=radii_range[0],
        maxRadius=radii_range[1]
    )

    circ_img = image.copy()
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(circ_img, (i[0], i[1]), i[2], (0, 255, 0), 2)

    # Harris Corner Detector
    corners = corner_peaks(corner_harris(gray), min_distance=10, threshold_rel=0.010)  # Increased min_distance, added threshold_rel
    corner_img = image.copy()
    for corner in corners:
        cv2.drawMarker(corner_img, tuple(corner[::-1]), (0, 0, 255), markerType=cv2.MARKER_CROSS)


    # Gabor Filters
    gabor_img = np.zeros_like(gray, dtype=np.float32)
    for theta in [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]:
        filt_real, _ = gabor(gray, frequency=0.6, theta=theta)
        gabor_img += np.abs(filt_real)
    gabor_img = np.clip(gabor_img, 0, 255).astype(np.uint8)
    gabor_img = cv2.merge([gabor_img, gabor_img, gabor_img])

    return circ_img, corner_img, gabor_img

def main():
    st.title("Retinal Feature Detection: Hemorrhage and Drusen")
    uploaded_file = st.file_uploader("Choose a fundus image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # CLAHE
        clahe_image = apply_clahe(image)

        # Detect features
        radii_range = (5, 20)
        circ_img, corner_img, gabor_img = detect_features(clahe_image, radii_range)

        # Display images
        st.image(clahe_image, caption="CLAHE Image", use_column_width=True)
        st.image(circ_img, caption="Circular Hough Transform", use_column_width=True)
        st.image(corner_img, caption="Harris Corner Detection", use_column_width=True)
        st.image(gabor_img, caption="Gabor Filter Response", use_column_width=True)

if __name__ == "__main__":
    main()


from pyngrok import ngrok

# Kill any existing streams
!pkill streamlit

# Authenticate ngrok
!ngrok authtoken 2i4KlnXgeuiLXrI639JnTkwif8c_3wEWjKjaUbiE3iUuP6Edj

# Create a public URL
ngrok_tunnel = ngrok.connect(addr='8501', proto='http')
print(f'Streamlit app will be accessible from this URL: {ngrok_tunnel.public_url}')

# Run the Streamlit app
!streamlit run app.py &>/dev/null&
