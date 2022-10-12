from classification import get_classification
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

if __name__ == '__main__':
    input_image = Image.open(r"S:\ds_portfolio_proj\clf_imageNet_mnV2\backend\data\Images\dog.jpg")
    # input_image.show()
    predictions =  get_classification(input_image,
                            r"S:\ds_portfolio_proj\clf_imageNet_mnV2\backend\data\ImageNET_classes.txt")
    # print(predictions)
    img_pred = ImageDraw.Draw(input_image)
    myFont = ImageFont.truetype('arial.ttf',40)
    for i in range(5):
        img_pred.text((input_image.size[0]*0.7, input_image.size[0]*0.05+40*i), '{:.2f}%  - '.format(predictions[i][1]*100)+str(predictions[i][0]), font=myFont, fill =(0,255,0)) #(5+50*i, 210-40*i, 10+40*i) - for dynamic color
 
    # Display edited image
    input_image.show()
    
    # Save the edited image
    input_image.save(r"S:\ds_portfolio_proj\clf_imageNet_mnV2\backend\data\Predictions\dog_pred.jpg")