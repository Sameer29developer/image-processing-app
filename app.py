import streamlit as st
import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage import exposure

st.set_page_config(page_title="Image Processing App", layout="wide")
st.title("🖼️ Comprehensive Image Processing App")

# File uploader
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Read image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    st.image(img_rgb, caption='Original Image', use_column_width=True)

    # Options
    st.sidebar.header("Choose Processing Operation")
    option = st.sidebar.selectbox("Select Task", [
        "Fourier Transform",
        "Histogram & Metrics",
        "Edge Detection",
        "Filtering",
        "Enhancement"
    ])

    if option == "Fourier Transform":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)

        fig, ax = plt.subplots()
        ax.imshow(magnitude_spectrum, cmap='gray')
        ax.set_title('Fourier Magnitude Spectrum')
        ax.axis('off')
        st.pyplot(fig)

    elif option == "Histogram & Metrics":
        channels = ('r', 'g', 'b')
        fig, ax = plt.subplots()
        for i, col in enumerate(channels):
            histr = cv2.calcHist([img_rgb], [i], None, [256], [0, 256])
            ax.plot(histr, color=col)
        ax.set_title("RGB Histogram")
        st.pyplot(fig)

        st.subheader("Image Statistics")
        st.write(f"Mean: {np.mean(img_rgb):.2f}")
        st.write(f"Standard Deviation: {np.std(img_rgb):.2f}")
        st.write(f"Min Pixel Value: {np.min(img_rgb)}")
        st.write(f"Max Pixel Value: {np.max(img_rgb)}")

    elif option == "Edge Detection":
        method = st.sidebar.radio("Choose Edge Detection Method", ("Canny", "Sobel", "Laplacian"))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if method == "Canny":
            edges = cv2.Canny(gray, 100, 200)
        elif method == "Sobel":
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
            edges = cv2.magnitude(sobelx, sobely)
        elif method == "Laplacian":
            edges = cv2.Laplacian(gray, cv2.CV_64F)

        st.image(np.uint8(np.abs(edges)), caption=f"{method} Edges", use_column_width=True)

    elif option == "Filtering":
        filter_type = st.sidebar.radio("Choose Filter", ("Gaussian", "Median", "Bilateral"))
        if filter_type == "Gaussian":
            ksize = st.sidebar.slider("Kernel Size", 1, 21, 5, step=2)
            filtered = cv2.GaussianBlur(img_rgb, (ksize, ksize), 0)
        elif filter_type == "Median":
            ksize = st.sidebar.slider("Kernel Size", 1, 21, 5, step=2)
            filtered = cv2.medianBlur(img_rgb, ksize)
        elif filter_type == "Bilateral":
            d = st.sidebar.slider("Diameter", 1, 20, 9)
            sigma_color = st.sidebar.slider("Sigma Color", 1, 100, 75)
            sigma_space = st.sidebar.slider("Sigma Space", 1, 100, 75)
            filtered = cv2.bilateralFilter(img_rgb, d, sigma_color, sigma_space)

        st.image(filtered, caption=f"{filter_type} Filtered", use_column_width=True)

    elif option == "Enhancement":
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        equalized = cv2.equalizeHist(gray)

        st.image(equalized, caption="Histogram Equalized Image", use_column_width=True)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 3))
        ax1.hist(gray.ravel(), 256, [0, 256])
        ax1.set_title('Original Histogram')
        ax2.hist(equalized.ravel(), 256, [0, 256])
        ax2.set_title('Equalized Histogram')
        st.pyplot(fig)
