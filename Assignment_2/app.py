import os
from flask import Flask, request, render_template, send_file, url_for
import cv2
import numpy as np
from PIL import Image
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def apply_filter(image_path, filter_name):
    # Read the image
    img = cv2.imread(image_path)
    
    if filter_name == 'grayscale':
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    elif filter_name == 'blur':
        return cv2.GaussianBlur(img, (9, 9), 0)
    elif filter_name == 'edge_detection':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.Canny(gray, 100, 200)
    elif filter_name == 'sharpen':
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        return cv2.filter2D(img, -1, kernel)
    elif filter_name == 'sepia':
        kernel = np.array([[0.272, 0.534, 0.131],
                          [0.349, 0.686, 0.168],
                          [0.393, 0.769, 0.189]])
        return cv2.transform(img, kernel)
    return img

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Apply filter if specified
            filter_name = request.form.get('filter', 'original')
            if filter_name != 'original':
                filtered_img = apply_filter(filepath, filter_name)
                output_filename = f'filtered_{filename}'
                output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
                cv2.imwrite(output_path, filtered_img)
                return render_template('index.html', 
                                     original_image=url_for('static', filename=f'uploads/{filename}'),
                                     filtered_image=url_for('static', filename=f'uploads/{output_filename}'))
            
            return render_template('index.html', 
                                 original_image=url_for('static', filename=f'uploads/{filename}'))
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host = '0.0.0.0', port = 5000)
