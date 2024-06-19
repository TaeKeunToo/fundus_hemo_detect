!pip install streamlit
!pip install pyngrok
!pip install numpy
!pip install opencv-python-headless
!pip install matplotlib




# new cell
%%writefile app.py
import streamlit as st
import numpy as np
import cv2
from matplotlib import pyplot as plt

def apply_clahe(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    # Convert the image to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Apply CLAHE to the L-channel
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    cl = clahe.apply(l)

    # Merge the CLAHE enhanced L-channel with the a and b channels
    limg = cv2.merge((cl, a, b))

    # Convert the image back to BGR color space
    enhanced_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced_image

def apply_frequency_filter(image_channel, filter_type, D0):
    dft = cv2.dft(np.float32(image_channel), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)

    rows, cols = image_channel.shape
    crow, ccol = rows // 2, cols // 2

    mask = np.zeros((rows, cols, 2), np.uint8)
    if filter_type == 'High-pass':
        mask[:crow-int(D0), :] = 1
        mask[crow+int(D0):, :] = 1
        mask[:, :ccol-int(D0)] = 1
        mask[:, ccol+int(D0):] = 1
    elif filter_type == 'Low-pass':
        mask[crow-int(D0):crow+int(D0), ccol-int(D0):ccol+int(D0)] = 1

    fshift = dft_shift * mask

    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    
    return img_back

st.title("Fundus Photography Preprocessing with CLAHE and Frequency Filters")
st.write("Upload a fundus image for preprocessing using CLAHE and apply frequency-based filters to each RGB channel.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = np.array(cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), cv2.IMREAD_COLOR))
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    
    clip_limit = st.slider("CLAHE Clip Limit", min_value=1.0, max_value=10.0, value=2.0, step=0.1)
    tile_grid_size = st.slider("CLAHE Tile Grid Size", min_value=1, max_value=20, value=8)
    
    filter_type = st.selectbox("Select frequency filter type", ["High-pass", "Low-pass"])
    D0 = st.slider("Frequency filter Cutoff frequency (D0)", min_value=10, max_value=100, value=30)
    
    if st.button('Apply CLAHE and Frequency Filters'):
        # Apply CLAHE
        enhanced_image = apply_clahe(image, clip_limit, (tile_grid_size, tile_grid_size))
        
        # Split image into R, G, B channels after CLAHE
        R, G, B = cv2.split(enhanced_image)

        # Apply frequency filter to each channel
        filtered_R = apply_frequency_filter(R, filter_type, D0)
        filtered_G = apply_frequency_filter(G, filter_type, D0)
        filtered_B = apply_frequency_filter(B, filter_type, D0)

        # Display results
        st.write("Filtered R Channel:")
        plt.figure(figsize=(5, 5))
        plt.imshow(filtered_R, cmap='gray')
        plt.axis('off')
        st.pyplot(plt)

        st.write("Filtered G Channel:")
        plt.figure(figsize=(5, 5))
        plt.imshow(filtered_G, cmap='gray')
        plt.axis('off')
        st.pyplot(plt)

        st.write("Filtered B Channel:")
        plt.figure(figsize=(5, 5))
        plt.imshow(filtered_B, cmap='gray')
        plt.axis('off')
        st.pyplot(plt)




# new cell

from pyngrok import ngrok

# Kill any existing streams
!pkill streamlit

# Authenticate ngrok
!ngrok authtoken 2hiKPETHiVnk0VPwWWN6Yx2ALYg_5H9tDUTsoRWZ3xFKPuLrq

# Create a public URL
ngrok_tunnel = ngrok.connect(addr='8501', proto='http')
print(f'Streamlit app will be accessible from this URL: {ngrok_tunnel.public_url}')

# Run the Streamlit app
!streamlit run app.py &>/dev/null&
