import io
import base64
import uvicorn
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image

app = FastAPI()
model = YOLO(r"./yolov11.pt")  # Load YOLOv11 model

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class_names = ["rsc", "looper", "thrips", "jassid", "rsm", "tmb", "healthy"]

# Pest recommendations
pest_recommendations = {
    "tmb": {
        "name": "Tea Mosquito Bug (TMB)",
        "symptoms": [
            "Brown or black feeding spots on shoots.",
            "Wilted or damaged shoots due to nymph feeding.",
            "Presence of eggs and early nymph stages inserted in shoots.",
        ],
        "control_methods": {
            "biological": [
                "Encourage natural predators like Sycanus collaris (Reduviid bug), Chrysoperla carnea, Mallada boninensis (Lacewings), Oxyopes spp. (Spiders), and praying mantis."
            ],
            "chemical": [
                "Use systemic insecticides alternately: Thiamethoxam 25 WG, Clothianidin 50 WDG, Thiacloprid 21.7% SC, and Neem extract (Azadirachtin 5% W/W)."
            ],
            "mechanical": [
                "Regularly remove infested shoots during plucking.",
                "Maintain a close plucking schedule to remove eggs and nymphs.",
                "Remove alternate host plants near plantations.",
                "Prune and skiff bushes during cold weather.",
            ],
        },
    },
    "rsm": {
        "name": "Red Spider Mite (RSM)",
        "symptoms": [
            "Presence of red spots and webbing on the underside of leaves.",
            "Leaves turn bronze or rusty red and fall off.",
            "Infestation is more severe in unshaded, waterlogged areas.",
        ],
        "control_methods": {
            "biological": [
                "Conserve predatory insects and mites like Phytoseiid mites (Amblyseius sp., Cunaxa sp.), Ladybird beetles (Stethorus sp., Scymnus sp.), and Lacewings (Mallada sp., Chrysopa sp.)."
            ],
            "chemical": [
                "Use acaricides alternately: Propargite 57 EC, Fenazaquin 10 EC, Spiromesifen 240 SC, and Hexythiazox 5.45 EC."
            ],
            "mechanical": [
                "Maintain shade trees at recommended spacing to reduce mite buildup.",
                "Remove alternate host plants near plantations.",
                "Improve drainage to prevent waterlogging.",
            ],
        },
    },
    "rsc": {
        "name": "Red Slug Caterpillar (RSC)",
        "symptoms": [
            "Feeding damage on young leaves.",
            "Defoliation of bushes during severe infestations.",
            "Presence of caterpillars with a distinctive red body and slug-like appearance.",
        ],
        "control_methods": {
            "biological": [
                "Encourage natural enemies like birds, parasitic wasps, and pathogenic fungi and bacteria in the soil."
            ],
            "chemical": [
                "Apply insecticides like Emamectin Benzoate 5% SG and Flubendiamide 20% WG."
            ],
            "mechanical": [
                "Manual collection and destruction of caterpillars.",
                "Prune and clean bushes to remove pupae from crevices.",
            ],
        },
    },
    "looper": {
        "name": "Looper Caterpillar",
        "symptoms": [
            "Defoliation of bushes due to feeding by caterpillars.",
            "Caterpillars are visible hanging from leaves using silken threads.",
            "Eggs laid in clusters on cracks of shade tree bark.",
        ],
        "control_methods": {
            "biological": [
                "Encourage natural enemies like Cotesia ruficrus (Parasitoid wasp), Sycanus collaris (Predatory bug), spiders (Oxyopes shweta), and entomopathogenic nematodes (Steinernema sp., Heterorhabditis sp.)."
            ],
            "chemical": [
                "Apply insecticides alternately: Emamectin Benzoate 5% SG, Quinalphos 25 EC, and Deltamethrin 10 EC."
            ],
            "mechanical": [
                "Manual removal of caterpillars, moths, and chrysalids.",
                "Light scrapping of shade tree bark to destroy eggs.",
                "Use light traps during the evening to attract and kill moths.",
            ],
        },
    },
    "thrips": {
        "name": "Thrips",
        "symptoms": [
            "Leaves show silvering and curling due to feeding.",
            "Leaf tips may turn yellowish or brown.",
            "Both adult and larval thrips can be found on leaves.",
        ],
        "control_methods": {
            "biological": [
                "Encourage natural predators like predatory thrips (Aeolothrips intermedius, Mymarothrips garuda), spiders, and dragonflies."
            ],
            "chemical": [
                "Use systemic insecticides alternately: Thiamethoxam 25 WG, Clothianidin 50 WDG, and Bifenthrin 8 SC."
            ],
            "mechanical": [
                "Use yellow sticky traps (45 cm wide) to attract and trap thrips.",
                "Maintain a shade level of 60% in the plantation.",
            ],
        },
    },
    "jassid": {
        "name": "Jassid",
        "symptoms": [
            "Yellowing and curling of leaf edges.",
            "Leaves show brown spots and withering in severe infestations.",
            "Both adults and nymphs feed on the underside of leaves.",
        ],
        "control_methods": {
            "biological": [
                "Conserve natural predators like ladybeetles (Stethorus sp., Scymnus sp.), predatory bugs (Anthocoris sp., Orius sp.), and lacewings (Chrysopa sp., Chrysoperla sp.)."
            ],
            "chemical": [
                "Use systemic insecticides alternately: Thiamethoxam 25 WG, Clothianidin 50 WDG, and Spirotetramat 15.31% OD."
            ],
            "mechanical": [
                "Use light traps and yellow sticky traps to monitor and control populations.",
                "Caustic wash the trunk and stir soil around the collar region to kill pupae.",
            ],
        },
    },
}


@app.post("/yolo-v11/")
async def predict(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_cv = np.array(image)
        image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

        results = model(image)[0]  # Get first result from YOLO inference
        predictions = []

        if results.boxes is not None and len(results.boxes.xyxy) > 0:
            for i in range(len(results.boxes.xyxy)):
                box = results.boxes.xyxy[i]
                label_index = int(results.boxes.cls[i])
                confidence_score = float(results.boxes.conf[i])
                prediction_label = class_names[label_index]

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
                    1,
                    (0, 255, 0),
                    2,
                )

                # Process segmentation mask if available
                if results.masks is not None and results.masks.xy:
                    mask = np.zeros(image_cv.shape[:2], dtype=np.uint8)
                    cv2.fillPoly(
                        mask, [np.array(results.masks.xy[i], dtype=np.int32)], 255
                    )

                    # Create colored mask overlay
                    colored_mask = np.zeros_like(image_cv, dtype=np.uint8)
                    colored_mask[:, :, 0] = mask
                    image_cv = cv2.addWeighted(image_cv, 1, colored_mask, 0.5, 0)

                # Fetch pest recommendations
                recommendation = pest_recommendations.get(prediction_label, {})

                predictions.append(
                    {
                        "prediction": prediction_label,
                        "confidence": confidence_score,
                        "bounding_box": bbox_coords,
                        "symptoms": (
                            recommendation.get("symptoms", ["No symptoms available."])
                            if recommendation
                            else ["No symptoms available."]
                        ),
                        "biological_control": (
                            recommendation.get("control_methods", {}).get(
                                "biological", ["No biological control available."]
                            )
                            if recommendation
                            else ["No biological control available."]
                        ),
                        "chemical_control": (
                            recommendation.get("control_methods", {}).get(
                                "chemical", ["No chemical control available."]
                            )
                            if recommendation
                            else ["No chemical control available."]
                        ),
                        "mechanical_control": (
                            recommendation.get("control_methods", {}).get(
                                "mechanical", ["No mechanical control available."]
                            )
                            if recommendation
                            else ["No mechanical control available."]
                        ),
                    }
                )

        if not predictions:
            return {"message": "No predictions found", "predictions": []}

        _, buffer = cv2.imencode(".jpg", image_cv)
        encoded_image = base64.b64encode(buffer).decode("utf-8")

        return {
            "predictions": predictions,
            "processed_image": encoded_image,
        }
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
