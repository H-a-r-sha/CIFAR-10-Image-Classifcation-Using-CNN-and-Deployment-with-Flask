import os
import base64
import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input

app = Flask(__name__)
model = load_model('model.pkl')  # Load your trained model here

@app.route('/', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        image_file = request.files['image']
        image_path = os.path.join('uploads', image_file.filename)
        image_file.save(image_path)
        prediction = predict(image_path)
        image_data = base64.b64encode(open(image_path, 'rb').read()).decode('utf-8')
        return render_template('index.html', prediction=prediction, image=image_data)
    return render_template('index.html')

def predict(image_path):
    img = image.load_img(image_path, target_size=(32, 32))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    predicted_class = classes[np.argmax(preds)]
    return predicted_class

if __name__ == '__main__':
    app.run(debug=True)
