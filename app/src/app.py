import sys
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
from io import BytesIO
import os
import pillow_avif # plugin for avif files
import streamlit as st
from streamlit_image_comparison import image_comparison
# getting the processing modules by adding the path of the folder to sys
# sys.path.insert(0, r'../../backend/src')

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import backend.src.classification as b



def main():
    # st.title("Classification BOT")
    st.set_page_config(page_title='Welcome', layout='centered')
    # adding style to the page
    html_temp = """
    <h1 style="color:white;text-align:center;"> Classification BOT </h4>
    <div style="background-color:green;padding:10px;border-radius:12px;">
    <h4 style="color:white;text-align:center;"> I help you in classifying the Images!!! </h4>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    # to get the interface for uploading, we can use easy streamlit commands
    st.write("##")
    up_file = st.file_uploader("please upload an image that needs to be classified", type= ['png', 'jpg', 'jpeg', 'avif'])
    if up_file is not None: ## to validate the file existence
        input_image = Image.open(up_file)
        pred_img = input_image.copy()
        predictions =  b.get_classification(input_image,
                            r"S:\ds_portfolio_proj\clf_imageNet_mnV2\backend\data\ImageNET_classes.txt")
        st.write("## Predictions?, scroll to the right")
        img_pred = ImageDraw.Draw(pred_img)
        myFont = ImageFont.truetype('arial.ttf',40)
        for i in range(5):
            img_pred.text((input_image.size[0]*0.7, input_image.size[0]*0.05+40*i), '{:.2f}%  - '.format(predictions[i][1]*100)+str(predictions[i][0]), font=myFont, fill =(0,255,0)) #(5+50*i, 210-40*i, 10+40*i) - for dynamic color
        # st.dataframe(req_PosRev_data[['ID', 'Text', 'Star']].head())
        buf = BytesIO()
        input_image.save(buf, format="PNG")
        byte_im = buf.getvalue()
        # st.write("predictions?, scroll to the right")
        image_comparison(
            img1=pred_img,
            img2=input_image,
            label1="predictions",
            label2="original",
            starting_position=1,
            in_memory=False
        )

        st.write("Click  to download the image with predictions as PNG file")
        st.download_button(
             label="Download",
             data=byte_im,
             file_name=str(predictions[0][0])+'.png',
             mime='image/png',
         )

if __name__ == '__main__':  
    main()
