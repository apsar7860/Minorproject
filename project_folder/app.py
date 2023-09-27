from flask import Flask, render_template, request, jsonify

from PIL import Image
import io
#python module used to detect the type of the image file
import imghdr
#binary to text encoding scheme that represeents in ascii string format
import base64
import numpy as np
import cv2

app = Flask(__name__)

def is_image(file):
    # Check if the file is a valid image file using the imghdr module
    file_data = file.read()
    if file_data:
        image_type = imghdr.what(None, h=file_data)
        return image_type is not None
    return False

def process_image(image_data, scaleFactor=1.1, minNeighbors=8):
    # Decode the base64-encoded image data
    image_decoded = base64.b64decode(image_data)
    np_arr = np.frombuffer(image_decoded, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    # Convert the image to grayscale for face detection
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Load the pre-trained face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=scaleFactor, minNeighbors=minNeighbors)

    # Draw rectangles around detected faces on the original image
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # Encode the processed image to base64 for displaying in the results
    _, encoded_image = cv2.imencode('.png', image)
    processed_image_data = base64.b64encode(encoded_image).decode('utf-8')

    return processed_image_data, len(faces)
def process_signature(signature_path):
    # Load the signature image
    signature_image = cv2.imread(signature_path)

    if signature_image is None:
        raise ValueError("Failed to load the signature image")

    # Convert the signature image to grayscale
    gray_signature = cv2.cvtColor(signature_image, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to enhance the signature
    processed_signature = cv2.adaptiveThreshold(gray_signature, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    # Return the processed signature
    return processed_signature
def detect_swapped_fields(image_filename, signature_filename):
    image_in_image_field = "image" in image_filename.lower() or "signature" not in image_filename.lower()
    signature_in_signature_field = "signature" in signature_filename.lower() or "image" not in signature_filename.lower()

    if image_in_image_field and signature_in_signature_field:
        return False  # No swapping needed

    return True  # Swap needed

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit_form', methods=['POST'])
def submit_form():
    try:
        name = request.form.get('name')
        age = request.form.get('age')
        dob = request.form.get('dob')
        adhar = request.form.get('adhar')
        address = request.form.get('address')
        app_number = request.form.get('app_number')
        image = request.files['image']
        signature = request.files['signature']

        image_filename = image.filename
        signature_filename = signature.filename

        # Detect if fields need swapping
        swapping_done = detect_swapped_fields(image_filename, signature_filename)

        # Swap the fields if needed
        if swapping_done:
            image, signature = signature, image

        if int(age) < 18:
            return "You must be 18 years or older to submit the form."

        # Resize images before encoding to base64
        max_width = 300
        max_height = 300
        image = Image.open(image)
        image.thumbnail((max_width, max_height))
        signature = Image.open(signature)
        signature.thumbnail((max_width, max_height))

        # Encode images to base64
        image_data = image_to_base64(image)
        signature_data = image_to_base64(signature)

        processed_image_data, num_faces = process_image(image_data)

        # Get the file extension from the original image filename
        image_extension = image_filename.split('.')[-1].lower()

        # Determine the image type for displaying in the results
        if image_extension == 'jpg':
            image_type = 'jpeg'
        else:
            image_type = image_extension

        # Determine the signature type for displaying in the results
        signature_type = imghdr.what(None, h=base64.b64decode(signature_data))

        return render_template('results.html', name=name, age=age, dob=dob, adhar=adhar, address=address,
                               app_number=app_number, image_data=processed_image_data, signature_data=signature_data,
                               image_type=image_type, signature_type=signature_type, swapping_done=swapping_done)
    except Exception as e:
        return "An error occurred: " + str(e)

def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")  # You can change the format to JPEG or other supported formats
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

if __name__ == '__main__':
    app.run(debug=True)