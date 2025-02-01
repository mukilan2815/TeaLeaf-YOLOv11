import os
import io
import base64
import uvicorn
import torch
import numpy as np
import cv2
from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
from PIL import Image

app = FastAPI()
model = YOLO(r"D:\Titan\Projects\d\yolov11.pt")

class_names = ["rsc", "looper", "thrips", "jassid", "rsm", "tmb", "healthy"]

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

        results = model(image)
        prediction_label = "No predictions"
        found_prediction = False

        for result in results:
            if result.masks and result.masks.xy:
                for i, mask in enumerate(result.masks.xy):
                    label = class_names[int(result.boxes.cls[i])]
                    prediction_label = label
                    found_prediction = True
                    break
            if found_prediction:
                break

        if prediction_label == "No predictions":
            return {"prediction": "No predictions"}

        recommendation = pest_recommendations.get(prediction_label, None)
        _, buffer = cv2.imencode(".jpg", image_cv)
        encoded_image = base64.b64encode(buffer).decode("utf-8")

        return {
            "prediction": prediction_label,
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
            "processed_image": encoded_image,
        }

    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
