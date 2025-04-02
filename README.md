# YOLO-v11 API Project Documentation 🚀

## Project Overview 🌟
The YOLO-v11 API project is a web-based application built with FastAPI that leverages the Ultralytics YOLO model for image analysis. When an image is submitted, the API performs object detection, draws bounding boxes, and applies segmentation masks (if available). Additionally, it fetches and returns pest management recommendations based on the prediction, making it an excellent tool for automated image analysis and decision support.

## Pre-requirements ✅
- **Python 3.x:** Ensure that Python is installed on your system 🐍.
- **Model File:** A valid YOLO-v11 model file must be present at `utils/yolov11.pt` 📁.
- **Dependencies:** All required Python packages should be installed via the `requirements.txt` file.

## Setup Instructions 🛠️
1. **Clone the Repository:**
    - Clone the project's repository to your local machine 🌐.
      ```
      git clone https://github.com/TitanNatesan/TeaLeaf-YOLOv11.git
      ```     
2. **Install Dependencies:**
    - Navigate to the project directory and run:
      ```
      pip install -r requirements.txt
      ```
    - This command installs necessary libraries such as FastAPI, uvicorn, numpy, OpenCV, Pillow, and the Ultralytics YOLO package.

3. **Configure the Project:**
    - Verify that the YOLO model file is correctly placed in the `utils` directory.
    - Review the project structure and adjust any file paths if needed.

## Running the Project ▶️
1. **Start the API Server:**
    - From the command-line interface, navigate to the project folder containing `yolo11.py` (typically under `src`).
    - Run the following command to start the FastAPI server:
      ```
      uvicorn src.yolo11:app --host 0.0.0.0 --port 8000
      ```
    - The server will now be running and accessible at `http://0.0.0.0:8000`.

2. **Accessing the API:**
    - Use the provided `index.html` in the `src` or `test` folder to test the API via a web browser.
    - Alternatively, use an API client like Postman or curl to send a POST request to:
      ```
      http://localhost:8000/yolo-v11/
      ```
    - The endpoint accepts an image file and returns a JSON response with the prediction, detection confidence, bounding box coordinates, pest recommendations (symptoms, biological, chemical, and mechanical controls), and a base64-encoded processed image.

## Additional Notes 📝
- **CORS Configuration:** The API is configured with CORS middleware to accept requests from any origin. Adjust this configuration as needed for your security policies.
- **Model Performance:** Make sure your system meets the computational requirements for efficient image processing and object detection.
- **Error Handling:** Basic error handling is implemented to provide meaningful error messages if image processing fails.

Enjoy setting up and extending the YOLO-v11 API project! Happy coding! 🎉