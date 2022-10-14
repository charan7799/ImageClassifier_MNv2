
import torch
from torch.nn import functional as F
import sys
import os

 
# getting the processing modules by adding the path of the folder to sys
# sys.path.insert(0, r'..\..\backend')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from preprocess import *
# from get_inferences import * 
# from get_inferences import *
from processing.preprocess import preprocess_image, get_inference

def get_classification(input_image, inference_file):
    # show_image(image_file)
    input_batch = preprocess_image(input_image)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load(r"backend\models\mobilenetv2.pth")
    model.eval()
    with torch.no_grad():
        output = model(input_batch.to(device))
    probabilities = F.softmax(output[0], dim=0)

    classification_categories = get_inference(probabilities, inference_file)

    
    return classification_categories
