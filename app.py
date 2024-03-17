from flask import Flask, render_template, request , jsonify
import tensorflow as tf
from keras.models import load_model
import os
from PIL import Image
import google.generativeai as genai


app = Flask(__name__,template_folder="templates")

model = load_model(
    os.path.join("model\model_20class_20ep_lastest_v1.h5"),
    compile=False,
)

datadict = {
    0: "BooPadPongali",
    1: "FriedChicken",
    2: "GaengKeawWan",
    3: "GoongObWoonSen",
    4: "HoyKraeng",
    5: "HoyLaiPrikPao",
    6: "Joke",
    7: "KaoMooDang",
    8: "KhaoMokGai",
    9: "KkaoKlukKaphi",
    10: "KorMooYang",
    11: "KuaKling",
    12: "LarbMoo",
    13: "MooSatay",
    14: "NamTokMoo",
    15: "PadPakBung",
    16: "PadThai",
    17: "Somtam",
    18: "TomKhaGai",
    19: "TomYumGoong",
}

datadict_th = {
    "BooPadPongali": "ปูผัดผงกระหรี่",
    "FriedChicken": "ไก่ทอด",
    "GaengKeawWan": "แกงเขียวหวาน",
    "GoongObWoonSen": "กุ้งอบวุ้นเส้น",
    "HoyKraeng": "หอยแครง",
    "HoyLaiPrikPao": "หอยลายผัดพริกเผา",
    "Joke": "โจ๊ก",
    "KaoMooDang": "ข้าวหมูแดง",
    "KhaoMokGai": "ข้าวหมกไก่",
    "KkaoKlukKaphi": "ข้าวคลุกกระปิ",
    "KorMooYang": "คอหมูย่าง",
    "KuaKling": "คั่วกลิ้ง",
    "LarbMoo": "ลาบหมู",
    "MooSatay": "หมูสะเต๊ะ",
    "NamTokMoo": "น้ำตกหมู",
    "PadPakBung": "ผัดผักบุ้ง",
    "PadThai": "ผัดไทย",
    "Somtam": "ส้มตำ",
    "TomKhaGai": "ต้มข่าไก่",
    "TomYumGoong": "ต้มยำกุ้ง",
}

def process_image(image):
    try:
        img = Image.open(image)
        if img.mode == "RGBA":
            img = img.convert("RGBA").convert("RGB")
        img = img.resize((224, 224))
        img_array = tf.convert_to_tensor(img, dtype=tf.float32)
        img_array = tf.expand_dims(img_array, 0)

        return img_array
    
    except Exception as e:
        print(f"An error occurred from process_image function: {e}")
        return None
    
    
def send_prompt_to_gemini(predicted_class_name):
    # api_key = os.getenv("APIKEYGEMINI")
    api_key = "AIzaSyDx3_F4z_zkO1Bq4t-8TiAK2XG96Z0I5gg"
    genai.configure(api_key=api_key)
    generation_config = {
        "temperature": 0.1,
        "top_p": 1,
        "top_k": 1,
        "max_output_tokens": 5000,
    }

    safety_settings = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
    ]

    try:
        model = genai.GenerativeModel(
            model_name="gemini-pro",
            generation_config=generation_config,
            safety_settings=safety_settings
        )

        prompt_parts = [
            f"Please provide a {predicted_class_name} with the following details: Name of the dish , List of ingredients with amounts , Step-by-step cooking method/instructions , Level of spiciness (mild, medium, hot, etc.), Approximate nutritional information (calories, protein, fat, carbs, etc.) per serving. I'm looking for an authentic and flavorful Thai recipe that covers all those components. If possible, please also mention any dietary restrictions or allergies the recipe can accommodate (vegetarian, gluten-free, nut-free, etc.). I want my response to be in markdown format, with Name of the dish using ### and other topics using ## and emoji to decorate the topics.",
        ]

        response = model.generate_content(prompt_parts)
        return response.text
    except Exception as e:
        print(f"Error occurred: {e}")
        return "Please try again gemini is reconnecting..."


@app.route('/predict', methods=['POST'])
def predict():
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    try:
        file = request.files['file']
        image_data = process_image(file)
        prediction = model.predict(image_data, use_multiprocessing=True)
        predicted_class = tf.argmax(prediction, axis=1)[0]
        predicted_class_name = datadict[int(predicted_class)]
        confidence = tf.reduce_max(prediction)
        confidence_percentage = int(confidence * 100)
        chat_response = send_prompt_to_gemini(predicted_class_name)



        return {"predicted_class_name": predicted_class_name, "confidence_percentage":confidence_percentage , "chat_response": chat_response}
    
    except Exception as e:
        return {"error_msg": e}

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=3000)