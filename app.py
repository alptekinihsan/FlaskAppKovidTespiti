
"""## İhsan ALPTEKİN
      16541526
      İKİNCİ ÖĞRETİM
"""

from flask import Flask, render_template, request, session, redirect, url_for, flash
import os
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import cv2
import numpy as np

KLASOR_YUKLEME = './flask app/assets/images'
UZANTILAR = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__, static_url_path='/assets',
            static_folder='./flask app/assets',
            template_folder='./flask app')
app.config['KLASOR_YUKLEME'] = KLASOR_YUKLEME


@app.route('/')
def root():
    return render_template('index.html')


@app.route('/index.html')
def index():
    return render_template('index.html')


@app.route('/contact.html')
def contact():
    return render_template('contact.html')


@app.route('/news.html')
def news():
    return render_template('news.html')


@app.route('/about.html')
def about():
    return render_template('about.html')


@app.route('/faqs.html')
def faqs():
    return render_template('faqs.html')


@app.route('/prevention.html')
def prevention():
    return render_template('prevention.html')


@app.route('/upload.html')
def upload():
    return render_template('upload.html')



@app.route('/upload_ct.html')
def upload_ct():
    return render_template('upload_ct.html')


@app.route('/uploaded_ct', methods=['POST', 'GET'])
def uploaded_ct():
    if request.method == 'POST':
        # dosya kontrolü
        if 'file' not in request.files:
            flash('Dosya seçilmedi')
            return redirect(request.url)
        file = request.files['file']
        # Butona boş değer dönüyor ise
        if file.filename == '':
            flash('Dosya seçilmedi')
            return redirect(request.url)
        if file:
            file.save(os.path.join(app.config['KLASOR_YUKLEME'], 'upload_ct.jpg'))


    vgg16_ct = load_model('models/covid_new_data1.h5')

    image = cv2.imread('./flask app/assets/images/upload_ct.jpg')  # dosya okuma
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # resimleri formatlama
    image = cv2.resize(image, (224, 224))  # boyut sınırlama
    image = np.array(image) / 255
    image = np.expand_dims(image, axis=0) # Diziyi tek boyuta indirgedik.


    vgg_deger = vgg16_ct.predict(image)
    itml = vgg_deger[0]
    print("Tarama Sonucu:")
    if itml[0] > 0.5:
        vgg_ct_deger = str('%.2f' % (itml[0] * 100) + '% COVID 19')
    else:
        vgg_ct_deger = str('%.2f' % ((1 - itml[0]) * 100) + '% COVİD 19 Değil')
    print(vgg_ct_deger)


    return render_template('results_ct.html', vgg_ct_deger=vgg_ct_deger)


if __name__ == '__main__':
    app.secret_key = ".."
    app.run()