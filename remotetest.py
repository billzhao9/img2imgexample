import requests
import json
import base64
from io import BytesIO
from PIL import Image
import datetime
now = datetime.datetime.now()
#import testconfig
#cerebrium deploy --hardware A10 controlnet-inpainting private-646e09130bfc8271bccb
url = rf"XXXXXXXX (REPLACE ME)"
headers = {
    "Authorization": "XXX  (REPLACE ME)",
    "Content-Type": "application/json"
}

img_name = "test.jpg"
with open(img_name, "rb") as f:
    bytes = f.read()
    encoded = base64.b64encode(bytes).decode('utf-8')



data = {
   
    'images_data': encoded,  
    "prompt": "mohawk style,  curved, red hair,anime, 2d, cute",
    "negative_prompt": "ugly, boring, bad anatomy, blurry, pixelated, trees, green, obscure, unnatural colors, poor lighting, dull, and unclear",
    "num_images_per_prompt": 1,
    "num_inference_steps": 25,
    "guidance_scale": 7.5,
    #"seed": 33,
}


date_string = now.strftime("%Y-%m-%d_%H-%M-%S")
response = requests.post(url, headers=headers, data=json.dumps(data))

print(response.json())
with open('output.txt', 'w+') as f:
    f.write(str(response.json()))



imgstr = response.json()["result"]

for i, img_str in enumerate(imgstr):
    # decode the base64-encoded string to bytes
    img_data = base64.b64decode(img_str)
    #print(img_data)
    #convert the bytes to a PIL image object
    img = Image.open(BytesIO(img_data))

    #print(img)
    # # save the image to disk with a unique filename
    filename = f"image_{i}_{date_string}.png"
    with open(filename, "wb") as f:
        img.save(f, format="PNG")
