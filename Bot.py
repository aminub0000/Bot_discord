import discord
import time
import tensorflow as tf
from discord.ext import commands
import numpy as np
import pandas as pd
from discord import Embed
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage
import os

from datetime import datetime
import cv2
# Initialize Firebase app with your credentials
cred = credentials.Certificate('************************')#token firebase
 
firebase_admin.initialize_app(cred, {
    'storageBucket': '********.appspot.com'
})
bucket = storage.bucket()

intents = discord.Intents.default()
intents.message_content = True
client = commands.Bot(command_prefix='!', intents=intents)
model = tf.keras.models.load_model('melanoma_detection_100v4.h5')

def process_predictions(predictions):
    classification = ["melanoma benign", "melanoma malignant", "pigmented benign keratosis"]
    response = ""
    for i, class_name in enumerate(classification):
        precision = round(predictions[0][i] * 100, 2)
        response += f"{class_name} precision: {precision}%\n"
    return response

import sys
def progressbar(it, prefix="", size=60, out=sys.stdout): # Python3.3+
    count = len(it)
    def show(j):
        x = int(size*j/count)
        print("{}[{}{}] {}/{}".format(prefix, "█"*x, " "*(size-x), j, count), 
                end='\r', file=out, flush=True)
    show(0)
    for i, item in enumerate(it):
        yield item
        show(i+1)
    print("\n", flush=True, file=out)

async def send_message(message, response, image_path=None):
    try:
        if image_path:
            with open(image_path, 'rb') as file:
                image = discord.File(file)
                await message.channel.send(response, file=image)
        else:
            await message.channel.send(response)
    except Exception as e:
        print("Error: ", e)

async def send_options(message):
    options = ["Dataset used for Bot AI development" ,"algorithm and architecture used for model", "The results of Bot AI", "Send an image for prediction"]
    option_string = "\n".join(f"\U0001F539 {option}" for option in options)
    response = f"Please select an option by typing the number option:\n{option_string}"
    await send_message(message, response)

async def handle_option_selection(message, option):
    if option == "My resultat training as an AI Bot" or option == 1:
        response = f"You selected option:cd;snsdjvkds {option}"
        await send_message(message, response)
    response = f"You selected option: {option}"
    await send_message(message, response)
    
@client.event
async def on_ready():
    print(f'{client.user} is running successfully')

@client.event
async def on_message(message):
    if message.author == client.user:
        return
    
    user_message = message.content.lower()

    if user_message == "/hi":
        await send_message(message, "Hi!")
        print('answer')
    if user_message == "/options":  
        await send_options(message)

    if user_message.isdigit():
        option = int(user_message)
        if option == 1 :
             await send_message(message, "The dataset used in this AI Bot was obtained from ISIC, which is a collection of skin cancer images. The dataset includes a total of 15,000 images from three classes (melanoma benign, melanoma benign, melanoma benign) of skin cancers."+
"MediaFire dataset link:https://www.mediafire.com/file/isnjv4g1k9mpibm/Dataset_ISIC_FOR_skin_cancer.rar/file")
        elif option == 2 :
            await send_message(message, "This diagram shows the structure used in CNN" , "C:\\Users\\Amine\\Desktop\\Nouveau dossier\\shema.png")
        elif option == 3 :
            data = {
    '': ['0', '1', '2', 'accuracy', 'macro avg', 'weighted'],

    'precision': ['0.89', '0.90', '0.91', 'NaN', '0.90', '0.90'],
    'recall': ['0.95', '0.84', '0.91', 'NaN', '0.90', '0.90'],
    'f1-score': ['0.92', '0.87', '0.91', '0.90', '0.90', '0.90'],
    'support': ['500', '500', '332', '1332', '1332', '1332']
        }

            df = pd.DataFrame(data)
            formatted_df = df.to_string(index=False)

            await send_message(message, "```" + formatted_df + "```")
        elif option == 3 :
            await send_message(message, "Send your image skin cancer")
    for attachment in message.attachments:
        if attachment.filename.endswith(('.jpg', '.jpeg', '.png')):
            image_path = 'C:\\Users\\Amine\\Desktop\\Nouveau dossier\\'
            await attachment.save(attachment.filename)
            #Load and preprocess the image
            image = tf.keras.preprocessing.image.load_img(attachment.filename, target_size=(128, 128))
            image = tf.keras.preprocessing.image.img_to_array(image)
            image = np.expand_dims(image, axis=0)
            image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
            # Make prediction using the model
            # Process the predictions and generate a response
            predictions = model.predict(image)
            response = process_predictions(predictions)
            
            max_index= 0
            classification = ["melanoma benign", "melanoma malignant", "pigmented benign keratosis"]
            max_index = np.argmax(predictions)
            predicted_class = classification[max_index]

            await send_message(message, response)
            # Send the response message
            
            current_datetime = datetime.now()
            formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
            blob = bucket.blob(message.author.display_name+" predicted as "+predicted_class+" at "+formatted_datetime)
            blob.upload_from_filename(os.path.join(image_path ,attachment.filename))
            
TOKEN = '**************************************'#token discord
client.run(TOKEN)
