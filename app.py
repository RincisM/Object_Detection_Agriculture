import streamlit as st
import cv2
import torch
import numpy as np

# Load YOLO model
model = torch.load("D:/mcass/semester2/deeplearning/anaconda3/envs/yolo_project/yolov8m_custom.pt")
model.eval()

# Function to perform object detection
def detect_objects(image):
    # Preprocess the image
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 640))  # Adjust size if needed
    img = img.astype(np.float32) / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)

    # Pass the image through the YOLO model
    with torch.no_grad():
        output = model(img)

    # Process YOLO output to get bounding boxes
    # (Assuming output contains bounding box predictions)
    # boxes = process_yolo_output(output)

    # Draw bounding boxes on the image
    image_with_boxes = image.copy()
    # for box in boxes:
    #     x, y, w, h = box
    #     x, y, w, h = int(x), int(y), int(w), int(h)
    #     cv2.rectangle(image_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return image_with_boxes

# Streamlit app
st.title("YOLO Object Detection App")

# Upload image through Streamlit
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read image
    image = cv2.imread(uploaded_file.name)

    # Perform object detection
    result_image = detect_objects(image)

    # Display result
    st.image(result_image, caption="Object Detection Result", use_column_width=True)
