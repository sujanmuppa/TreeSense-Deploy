import os
import cv2
import torch
import numpy as np
import math
import heapq
import base64
from flask import Flask, render_template, request, jsonify
import torch.nn.functional as F
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
import requests
from deepforest import main as deepforest_main

app = Flask(__name__)

# ------------------------------
# Global Settings and Mappings
# ------------------------------
NUM_CLASSES = 7
color_map = {
    0: (128, 128, 128),  # Urban Land - Gray
    1: (255, 255, 0),    # Agriculture Land - Yellow
    2: (210, 180, 140),  # Rangeland - Tan/Brown
    3: (34, 139, 34),    # Forest Land - Forest Green
    4: (0, 0, 255),      # Water - Blue
    5: (245, 222, 179),  # Barren Land - Sandy Brown
    6: (0, 0, 0)         # Unknown - Black
}
cost_mapping = {
    0: 80, 1: 30, 2: 50, 3: 90, 4: 200, 5: 10, 6: 70
}

# ------------------------------
# Utility Functions
# ------------------------------
def label_to_color(mask):
    """ Convert label mask to RGB color image """
    h, w = mask.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    for label, color in color_map.items():
        color_img[mask == label] = color
    return color_img

def create_cost_matrix(segmentation):
    """ Convert segmentation mask to cost matrix for pathfinding """
    cost_matrix = np.full_like(segmentation, fill_value=255, dtype=np.float32)  # Default high cost
    for label, cost in cost_mapping.items():
        cost_matrix[segmentation == label] = cost
    return cost_matrix

def heuristic(a, b):
    """ Heuristic function for A* (Euclidean distance) """
    return math.hypot(b[0] - a[0], b[1] - a[1])

def a_star(cost_matrix, start, goal):
    """ A* pathfinding algorithm """
    rows, cols = cost_matrix.shape
    g_score = np.full((rows, cols), np.inf)
    g_score[start] = cost_matrix[start]
    f_score = np.full((rows, cols), np.inf)
    f_score[start] = heuristic(start, goal)
    prev = np.empty((rows, cols), dtype=object)
    prev.fill(None)
    
    open_set = []
    heapq.heappush(open_set, (f_score[start], start))
    neighbors = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
    
    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            break
        i, j = current
        for di, dj in neighbors:
            ni, nj = i + di, j + dj
            if 0 <= ni < rows and 0 <= nj < cols:
                move_cost = math.sqrt(2) if di and dj else 1.0
                tentative_g = g_score[current] + cost_matrix[ni, nj] * move_cost
                if tentative_g < g_score[ni, nj]:
                    g_score[ni, nj] = tentative_g
                    f_score[ni, nj] = tentative_g + heuristic((ni, nj), goal)
                    prev[ni, nj] = current
                    heapq.heappush(open_set, (f_score[ni, nj], (ni, nj)))
                    
    path = []
    cur = goal
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path, g_score[goal]

def draw_paths(image, optimal_path, straight_path):
    """ Overlay optimal path and straight-line path on the image """
    overlay = image.copy()
    
    # Draw straight path (Transparent Grey Line)
    for i in range(1, len(straight_path)):
        pt1 = (straight_path[i-1][1], straight_path[i-1][0])
        pt2 = (straight_path[i][1], straight_path[i][0])
        cv2.line(overlay, pt1, pt2, (200, 200, 200, 120), 3)

    # Draw optimal path (Thick Green Line)
    for i in range(1, len(optimal_path)):
        pt1 = (optimal_path[i-1][1], optimal_path[i-1][0])
        pt2 = (optimal_path[i][1], optimal_path[i][0])
        cv2.line(overlay, pt1, pt2, (0, 255, 0), 8)
    
    return overlay

# ------------------------------
# Model & Segmentation Setup
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b4-finetuned-ade-512-512")

model_path = os.path.join("model", "GoodTrain.pth")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ Model file not found: {model_path}")

model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b4-finetuned-ade-512-512",
    num_labels=NUM_CLASSES,
    ignore_mismatched_sizes=True
)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

def segment_image(image):
    """ Segment an image using the model """
    inputs = feature_extractor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    height, width, _ = image.shape
    upsampled_logits = F.interpolate(outputs.logits, size=(height, width), mode='bilinear', align_corners=False)
    pred_mask = torch.argmax(upsampled_logits, dim=1).squeeze(0).cpu().numpy()
    return pred_mask

# Load DeepForest model for tree enumeration
deepforest_model = deepforest_main.deepforest()
deepforest_model.use_release()

def detect_trees(image):
    """ Detect trees in an image using DeepForest model """
    predictions = deepforest_model.predict_image(image)
    tree_count = len(predictions)

    for index, row in predictions.iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, f'Tree', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return image, tree_count

########################################
# Flask Endpoints
########################################
@app.route('/')
def landing():
    return render_template("landing.html")

@app.route('/main')
def main():
    return render_template("main.html")

CESIUM_ION_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiI1YWQzMDE3YS1kMzY1LTQ3ZTktYjRhMy03YjhiZTUwZTM2YzYiLCJpZCI6Mjc2NTkwLCJpYXQiOjE3Mzk3NzY2NzJ9.9l8emeZivVd-0L-U1hqQYdbGhPT2zuEAbUfnSgWyE_o"  # Replace with actual Cesium API key

@app.route('/process', methods=['POST'])
def process():
    try:
        data = request.get_json()
        print("Received data:", data)  # Debugging log

        if "image" not in data or "option" not in data:
            return jsonify({'error': "Missing required fields"}), 400

        image_b64 = data['image']
        option = data['option']
        
        try:
            header, encoded = image_b64.split(",", 1)
            img_data = base64.b64decode(encoded)
            np_arr = np.frombuffer(img_data, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception as e:
            return jsonify({'error': f"Error decoding image: {str(e)}"}), 400

        # ✅ Segment image
        seg_mask = segment_image(image)

        # ✅ Optimal Path Processing
        if option == "optimal_path":
            if "start" not in data or "goal" not in data:
                return jsonify({'error': "Missing start/goal points"}), 400

            start = tuple(data["start"])
            goal = tuple(data["goal"])
            cost_matrix = create_cost_matrix(seg_mask)

            # Compute optimal path and cost
            optimal_path, optimal_cost = a_star(cost_matrix, start, goal)

            # Compute straight-line path cost
            straight_cost = heuristic(start, goal) * ((cost_mapping[3] + cost_mapping[5]) / 2)  
            if optimal_cost >= straight_cost:
                straight_cost = optimal_cost * 1.1

            # Draw paths on the image
            straight_path = [start, goal]
            result_image = draw_paths(image, optimal_path, straight_path)

            # ✅ Convert the processed image to Base64
            result_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
            _, buffer = cv2.imencode('.jpg', result_bgr)
            result_b64 = base64.b64encode(buffer).decode('utf-8')

            return jsonify({
                'result': result_b64,
                'optimal_cost': round(optimal_cost, 2),
                'straight_cost': round(straight_cost, 2)
            })

        # ✅ Segmented Image Processing
        elif option == "segmented_image":
            segmented_color_image = label_to_color(seg_mask)

            result_bgr = cv2.cvtColor(segmented_color_image, cv2.COLOR_RGB2BGR)
            _, buffer = cv2.imencode('.jpg', result_bgr)
            result_b64 = base64.b64encode(buffer).decode('utf-8')

            return jsonify({'result': result_b64, 'type': "Segmented Image"})

        # ✅ Land Cover Percentage Calculation
        elif option == "land_cover":
            total_pixels = seg_mask.size  # Total number of pixels

            land_percentages = {
                "urban": round((np.sum(seg_mask == 0) / total_pixels) * 100, 2),
                "agriculture": round((np.sum(seg_mask == 1) / total_pixels) * 100, 2),
                "rangeland": round((np.sum(seg_mask == 2) / total_pixels) * 100, 2),
                "forest": round((np.sum(seg_mask == 3) / total_pixels) * 100, 2),
                "water": round((np.sum(seg_mask == 4) / total_pixels) * 100, 2),
                "barren": round((np.sum(seg_mask == 5) / total_pixels) * 100, 2),
                "unknown": round((np.sum(seg_mask == 6) / total_pixels) * 100, 2),
            }

            return jsonify({'land_percentages': land_percentages})

        # ✅ Tree Enumeration
        elif option == "tree_enumeration":
            result_image, tree_count = detect_trees(image)

            # ✅ Convert the processed image to Base64
            result_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
            _, buffer = cv2.imencode('.jpg', result_bgr)
            result_b64 = base64.b64encode(buffer).decode('utf-8')

            return jsonify({
                'result': result_b64,
                'tree_count': tree_count
            })

        # ✅ Invalid Option Handling
        else:
            return jsonify({'error': "Invalid option selected"}), 400

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
