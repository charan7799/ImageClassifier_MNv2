from torchvision import transforms

from torch import topk
from matplotlib import pyplot as plt


    

def preprocess_image(input_image):

    data_transforms = transforms.Compose([transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    # input_image = Image.open(file)
    input_tensor = data_transforms(input_image)
    input_batch = input_tensor.unsqueeze(0)

    return input_batch


def get_inference(prob, inference_file):

    with open(inference_file, "r") as f:
        categories = [s.strip() for s in f.readlines()]
    top5_prob, top5_catid = topk(prob, 5)

    inferences = []

    for i in range(top5_prob.size(0)):
        inferences.append([categories[top5_catid[i]], top5_prob[i].item()])
    return inferences