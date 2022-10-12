import sys
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from io import BytesIO
import streamlit as st
# getting the processing modules by adding the path of the folder to sys
sys.path.insert(0, r'S:\ds_portfolio_proj\clf_imageNet_mnV2\backend\src')

from classification import get_classification
input_image = Image.open(r"S:\ds_portfolio_proj\clf_imageNet_mnV2\backend\data\dog.jpg")
    # input_image.show()
predictions =  get_classification(input_image,
                            r"S:\ds_portfolio_proj\clf_imageNet_mnV2\backend\data\ImageNET_classes.txt")


def main():
    st.title("Image Classifier")
    # adding style to the page
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h4 style="color:white;text-align:center;">Image Classifier for Images in ImageNET- webApp </h4>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    # to get the interface for uploading, we can use easy streamlit commands
    up_file = st.file_uploader("please upload an image", type= ['png', 'jpg', 'jpeg'])
    if up_file is not None: ## to validate the file existence
        input_image = Image.open(up_file)
        predictions =  get_classification(input_image,
                            r"S:\ds_portfolio_proj\clf_imageNet_mnV2\backend\data\ImageNET_classes.txt")
        st.write("## predictions for the given image are")
        img_pred = ImageDraw.Draw(input_image)
        myFont = ImageFont.truetype('arial.ttf',25)
        for i in range(5):
            img_pred.text((800, 10+40*i), '{:.2f}%  - '.format(predictions[i][1]*100)+str(predictions[i][0]), font=myFont, fill =(5+50*i, 210-40*i, 10+40*i))
        # st.dataframe(req_PosRev_data[['ID', 'Text', 'Star']].head())
        st.image(input_image, caption='predictions for the given image')
        st.write("Click  to download the image as JPG file")
        
        buf = BytesIO()
        input_image.save(buf, format="JPEG")
        byte_im = buf.getvalue()
        st.download_button(
             label="Download",
             data=byte_im,
             file_name=str(predictions[0][0])+'.jpg',
             mime='image/jpg',
         )

if __name__ == '__main__':  
    main()