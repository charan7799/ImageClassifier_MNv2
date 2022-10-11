
import torch
from torch.nn import functional as F
import sys


 
# getting the processing modules by adding the path of the folder to sys
sys.path.insert(0, r'S:\ds_portfolio_proj\clf_imageNet_mnV2\backend')

# from preprocess import *
# from get_inferences import * 
# from get_inferences import *
from processing import preprocess_image, get_inference

def get_classification(input_image, inference_file):
    # show_image(image_file)
    input_batch = preprocess_image(input_image)
    model = torch.load(r"S:\ds_portfolio_proj\clf_imageNet_mnV2\backend\models\mobilenetv2.pth")
    model.eval()
    with torch.no_grad():
        output = model(input_batch.cuda())
    probabilities = F.softmax(output[0], dim=0)

    classification_categories = get_inference(probabilities, inference_file)

    
    return classification_categories
