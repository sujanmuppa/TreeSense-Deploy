import os
import cv2
import torch
import numpy as np
import math
import heapq
import base64
from io import BytesIO
from flask import Flask, render_template, request, jsonify
import torch.nn.functional as F
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation

app = Flask(__name__)

# ------------------------------
# Global Settings and Mappings
# ------------------------------
NUM_CLASSES = 7
# Color mapping for visualization.
color_map = {
    0: (0, 255, 255),      # urban_land
    1: (255, 255, 0),      # agriculture_land
    2: (255, 0, 255),      # rangeland
    3: (0, 255, 0),        # forest_land
    4: (0, 0, 255),        # water
    5: (255, 255, 255),    # barren_land
    6: (0, 0, 0)           # unknown
}
# Cost mapping for path planning.
cost_mapping = {
    0: 1,
    1: 2,
    2: 3,
    3: 10,
    4: 5,
    5: 2,
    6: 5
}

# ------------------------------
# Utility Functions
# ------------------------------
def label_to_color(mask, color_map):
    h, w = mask.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    for label, color in color_map.items():
        color_img[mask == label] = color
    return color_img

def create_cost_matrix(segmentation, cost_mapping):
    cost_matrix = np.zeros_like(segmentation, dtype=np.float32)
    for label, cost in cost_mapping.items():
        cost_matrix[segmentation == label] = cost
    return cost_matrix

def heuristic(a, b):
    return math.hypot(b[0]-a[0], b[1]-a[1])

def a_star(cost_matrix, start, goal):
    rows, cols = cost_matrix.shape
    g_score = np.full((rows, cols), np.inf)
    g_score[start] = cost_matrix[start]
    f_score = np.full((rows, cols), np.inf)
    f_score[start] = heuristic(start, goal)
    prev = np.empty((rows, cols), dtype=object)
    prev.fill(None)
    
    open_set = []
    heapq.heappush(open_set, (f_score[start], start))
    # 8-connected neighbors.
    neighbors = [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(-1,1),(1,-1),(1,1)]
    
    while open_set:
        current_f, current = heapq.heappop(open_set)
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

def overlay_path_on_image(image, path):
    image_copy = image.copy()
    for (i, j) in path:
        cv2.circle(image_copy, (j, i), 2, (255, 0, 0), -1)
    for idx in range(1, len(path)):
        pt1 = (path[idx-1][1], path[idx-1][0])
        pt2 = (path[idx][1], path[idx][0])
        cv2.line(image_copy, pt1, pt2, (255, 0, 0), 2)
    return image_copy

# ------------------------------
# Model & Segmentation Setup
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
feature_extractor = SegformerFeatureExtractor.from_pretrained("nvidia/segformer-b4-finetuned-ade-512-512")
model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b4-finetuned-ade-512-512",
    num_labels=NUM_CLASSES,
    ignore_mismatched_sizes=True
)
model.load_state_dict(torch.load(os.path.join("model", "GoodTrain.pth"), map_location=device))
model.to(device)
model.eval()

def segment_image(image):
    inputs = feature_extractor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
    height, width, _ = image.shape
    upsampled_logits = F.interpolate(outputs.logits, size=(height, width), mode='bilinear', align_corners=False)
    pred_mask = torch.argmax(upsampled_logits, dim=1).squeeze(0).cpu().numpy()
    return pred_mask

########################################
# Flask Endpoints
########################################
@app.route('/')
def landing():
    return render_template("landing.html")

@app.route('/main')
def main():
    return render_template("main.html")

@app.route('/process', methods=['POST'])
def process():
    data = request.get_json()
    image_b64 = data['image']
    start = tuple(data['start'])
    goal = tuple(data['goal'])
    
    header, encoded = image_b64.split(",", 1)
    img_data = base64.b64decode(encoded)
    np_arr = np.frombuffer(img_data, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    seg_mask = segment_image(image)
    cost_matrix = create_cost_matrix(seg_mask, cost_mapping)
    path, total_cost = a_star(cost_matrix, start, goal)
    result_image = overlay_path_on_image(image, path)
    result_image_bgr = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.jpg', result_image_bgr)
    result_b64 = base64.b64encode(buffer).decode('utf-8')
    
    return jsonify({'result': result_b64, 'cost': total_cost})

if __name__ == '__main__':
    app.run(debug=True)
