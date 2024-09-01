from transformers import DetrImageProcessor, DetrForObjectDetection
import torch
from PIL import Image, ImageDraw, ImageFont
import requests
import json

import cv2
import random
from io import BytesIO


# you can specify the revision tag if you don't want the timm dependency
model_name = "facebook/detr-resnet-101"

processor = DetrImageProcessor.from_pretrained(model_name, revision="no_timm")
model = DetrForObjectDetection.from_pretrained(model_name, revision="no_timm")


font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
font_thickness = 1


def predict(url):
    if url.split(":")[0] == "http":
        print('http')
        image = Image.open(requests.get(url, stream=True).raw)
    else:
        print('local photo')
        image = Image.open(url)

    inputs = processor(images=image, return_tensors="pt")
    outputs = model(**inputs)

    # convert outputs (bounding boxes and class logits) to COCO API
    # let's only keep detections with score > 0.9
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]
    object_detected = []

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        object = {
            "label" : str(model.config.id2label[label.item()]),
            "confidence" : round(score.item(), 3),
            "bounding_box" : box,
        }
        object_detected.append(object)
    
    return {
        "status": "OK",
        "process_image" : f"{url}",
        "object_detected" : object_detected
    }


def load_image(filename):
    if filename.startswith("http"):
        print('Loading image from URL')
        try:
            response = requests.get(filename)
            image_data = BytesIO(response.content)
            image = Image.open(image_data)
        except Exception as e:
            raise RuntimeError(f"Failed to load image from URL: {e}")
    else:
        print('Loading image from local file')
        image = Image.open(filename)
    
    return image


def draw_boxes(image, json):
    boxes = json.get("object_detected", [])
    draw = ImageDraw.Draw(image)
    
    # You may need to specify a font for drawing text
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except IOError:
        font = ImageFont.load_default()

    print(boxes)
    for box in boxes:
        
        bbox = box["bounding_box"]
        label = box["label"]
        confidence = box["confidence"]

        color = tuple(random.randint(0, 255) for _ in range(3))
        thickness = 2

        # Extracting coordinates for the bounding box
        x_min, y_min, x_max, y_max = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])

        # Drawing the bounding box
        draw.rectangle([x_min, y_min, x_max, y_max], outline=color, width=thickness)

        # Forming text
        text = f"{label}: {confidence:.2f}"

        # Determining text size and position
        text_bbox = draw.textbbox((x_min, y_min), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_x = x_min
        text_y = y_min - text_height - 10

        # Drawing the background for the text
        draw.rectangle([text_x, text_y, text_x + text_width, text_y + text_height], fill=color)

        # Displaying text
        draw.text((text_x, text_y), text, font=font, fill=(0, 0, 0))

    return image


if __name__ == "__main__":
    url = r"D:/Ai/PetProject/findobject/Я.jpg"
    json = predict(url)
    image = load_image(url)
    image_with_boxes = draw_boxes(image, json)
    image_with_boxes.show()
    # render_detections(json)