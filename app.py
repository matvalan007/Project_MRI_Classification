from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
import cv2
import numpy as np

my_model = load_model('model_mri.h5')

app = Flask(__name__)


@app.route('/', methods=['GET'])
def main():
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def home():
    image = request.files['img']

    path = "./image/" + image.filename
    image.save(path)

    image1 = load_img(path)

    frame = img_to_array(image1)
    frame=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY )
    image_1_resize = cv2.resize(frame,(256,256))

    ans = np.dstack([image_1_resize,image_1_resize,image_1_resize])
    ans = np.expand_dims(ans, axis = 0)

    pred = my_model.predict(ans)
    return render_template('after.html', data = pred)


if __name__ == "__main__":
    app.run(port = 3000, debug = True)