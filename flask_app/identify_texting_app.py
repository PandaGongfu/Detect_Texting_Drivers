import os
# We'll render HTML templates and access data sent by POST
# using the request object from flask. Redirect and url_for
# will be used to redirect the user once the upload is done
# and send_from_directory will help us to send/show on the
# browser the file that the user just uploaded
from flask import Flask, render_template, request, redirect, url_for, jsonify, send_from_directory
from werkzeug import secure_filename
import cv2
import numpy as np
from sklearn.externals import joblib

# Initialize the Flask application
app = Flask(__name__)

# This is the path to the upload directory
app.config['UPLOAD_FOLDER'] = 'images/'
# These are the extension that we are accepting to be uploaded
app.config['ALLOWED_EXTENSIONS'] = set(['png', 'jpg', 'jpeg', 'JPEG', 'gif'])

# For a given file, return whether it's an allowed type or not
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in app.config['ALLOWED_EXTENSIONS']

# Classify drivers
def predict_image(filename):
    # fd = open(os.path.join(app.config['UPLOAD_FOLDER'], filename), 'rb')
    # image_str = fd.read()
    # image_np_arr = np.fromstring(image_str, np.uint8)
    # color_image = cv2.imdecode(image_np_arr, cv2.IMREAD_COLOR)
    color_image = cv2.imread(os.path.join(app.config['UPLOAD_FOLDER'], filename))
    sift_features = create_sift_features(color_image)

    cluster_model = joblib.load('models/cluster_model_final.pkl')
    class_model = joblib.load('models/classification_model_final.pkl')

    clustered_words = cluster_model.predict(sift_features)
    bow_hist = np.array([np.bincount(clustered_words, minlength=cluster_model.n_clusters)])
    return class_model.predict(bow_hist)[0], class_model.predict_proba(bow_hist)[0]

# create sift features
def create_sift_features(c_image):
    sift = cv2.xfeatures2d.SIFT_create()
    g_image = cv2.cvtColor(c_image, cv2.COLOR_BGR2GRAY)
    _, desc = sift.detectAndCompute(g_image, None)
    return desc

# This route will show a form to perform an AJAX request
# jQuery is loaded to execute the request and update the
# value of the operation
@app.route('/')
def index():
    return render_template('index.html')

# Route that will process the file upload
@app.route('/upload', methods=['POST'])
def upload():
    # Get the name of the uploaded file
    file = request.files['file']
    # Check if the file is one of the allowed types/extensions
    if file and allowed_file(file.filename):
        # Make the filename safe, remove unsupported chars
        filename = secure_filename(file.filename)
        # Move the file form the temporal folder to
        # the upload folder we setup
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # Redirect the user to the uploaded_file route, which
        # will basicaly show on the browser the uploaded file
        return redirect(url_for('uploaded_file',
                                filename=filename))

# This route is expecting a parameter containing the name
# of a file. Then it will locate that file on the upload
# directory and show it on the browser, so if the user uploads
# an image, that image is going to be shown after the upload
@app.route('/images/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

@app.route('/<filename>')
def send_image(filename):
    return send_from_directory("images", filename)

# Route that will process the file upload
@app.route('/classify', methods=['POST'])
def classify():
    data = request.json
    filename = data['filename']

    # Check if the file is one of the allowed types/extensions
    if allowed_file(filename):
        # Make the filename safe, remove unsupported chars
        filename = secure_filename(filename)

        # classify image
        driver_class, driver_prob = predict_image(filename)
        if  driver_class == 'texting':
            res_txt = 'Danger! Driver is texting!'+'   (Prob: '+str(1-driver_prob[0])+')'
        else:
            res_txt = 'That\'s a safe driver :-)'+'    (Prob: '+str(driver_prob[0])+')'

    return jsonify({'img': filename, 'res_txt': res_txt})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5566, debug=True)
