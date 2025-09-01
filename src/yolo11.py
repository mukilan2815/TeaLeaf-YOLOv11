import io
import base64
import uvicorn
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
from utils.rec import pest_recommendations

app = FastAPI()
model = YOLO(r"utils/master.pt")  # Load YOLOv11 model
print("Model loaded successfully!")  # Add this line
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

@app.middleware("http")
async def log_requests(request: Request, call_next):
    print(f"Incoming request: {request.method} {request.url}")
    try:
        response = await call_next(request)
        print(f"Response status: {response.status_code}")
        return response
    except Exception as e:
        print(f"Error processing request: {str(e)}")
        raise

class_names = [
"Looper_Mild",
"Looper_Moderate",
"Lopper_Severe",
"RSC_Mild",
"RSC_Moderate",
"RSC_Severe",
"RSM_Minor",
"RSM_Moderate",
"RSM_Severe",
"TGL_Mild",
"TGL_Moderate",
"TGL_Severe",
"TMB_Mild",
"TMB_Moderate",
"TMB_Severe",
"Thrips_Mild",
"Thrips_Moderate",
"Thrips_Severe"]


@app.post("/yolo-v11/")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_cv = np.array(image)
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

        results = model(image)
        prediction_label = "No predictions"
        confidence_score = 0.0
        bbox_coords = []
        found_prediction = False

        for result in results:
            if result.boxes:
                for i, box in enumerate(result.boxes.xyxy):
                    label_index = int(result.boxes.cls[i])
                    confidence_score = float(result.boxes.conf[i])
                    prediction_label = class_names[label_index]
                    found_prediction = True

                    # Bounding box
                    x1, y1, x2, y2 = map(int, box)
                    bbox_coords = [x1, y1, x2, y2]
                    cv2.rectangle(image_cv, (x1, y1), (x2, y2), (0, 255, 0), 4)

                    # Label
                    label_text = f"{prediction_label} ({confidence_score:.2f})"
                    cv2.putText(
                        image_cv,
                        label_text,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        4,  # increased font scale for a bigger label
                        (0, 255, 0),
                        10,  # increased thickness for a bolder font weight
                    )

                    # Process segmentation mask if available
                    if result.masks and result.masks.xy:
                        mask = np.zeros(image_cv.shape[:2], dtype=np.uint8)
                        cv2.fillPoly(
                            mask, [np.array(result.masks.xy[i], dtype=np.int32)], 255
                        )

                        # Create colored mask overlay (semi-transparent blue)
                        colored_mask = np.zeros_like(image_cv, dtype=np.uint8)
                        colored_mask[:, :, 0] = mask  # Blue channel

                        # Blend mask with image
                        alpha = 0.5
                        image_cv = cv2.addWeighted(image_cv, 1, colored_mask, alpha, 0)

                    break  # Process only the first detected object

            if found_prediction:
                break

        if not found_prediction:
            return {
                "prediction": "No predictions",
                "confidence": None,
                "bounding_box": None,
                "symptoms": "No symptoms available.",
                "biological_control": "No biological control available.",
                "chemical_control": "No chemical control available.",
                "mechanical_control": "No mechanical control available.",
                "processed_image": None,
            }

        _, buffer = cv2.imencode(".jpg", image_cv)
        encoded_image = base64.b64encode(buffer).decode("utf-8")

        # Fetch pest recommendations
        recommendation = pest_recommendations.get(prediction_label, None)

        return {
            "prediction": prediction_label,
            "confidence": confidence_score,
            "bounding_box": bbox_coords,
            "symptoms": (
                recommendation["symptoms"]
                if recommendation
                else "No symptoms available."
            ),
            "biological_control": (
                recommendation["control_methods"]["biological"]
                if recommendation
                else "No biological control available."
            ),
            "chemical_control": (
                recommendation["control_methods"]["chemical"]
                if recommendation
                else "No chemical control available."
            ),
            "mechanical_control": (
                recommendation["control_methods"]["mechanical"]
                if recommendation
                else "No mechanical control available."
            ),
            "processed_image": encoded_image,  # Image with bounding box and segmentation mask
        }

    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
