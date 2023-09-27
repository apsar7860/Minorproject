import cv2
from flask import Flask, render_template, request, jsonify, send_from_directory

# Define a global variable to store the processed image
processed_image = None

def process_image(image_path, scaleFactor=1.1, minNeighbors=8):
    # Load the image from the provided path
    image = cv2.imread(image_path)

    # Convert the image to grayscale for face detection
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Load the pre-trained face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml')

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=scaleFactor, minNeighbors=minNeighbors)

    # Draw rectangles around detected faces on the original image
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)

    return image, len(faces)

def main():
    global processed_image  # Access the global variable
    # Prompt the user to enter the image file name or path
    image_path = input("Enter the image file name or path: ")

    try:
        # Process the image and get the number of detected faces
        processed_image, num_faces = process_image(image_path)

        # Display the processed image with rectangles around detected faces
        cv2.imshow('Processed Image with Faces', processed_image)

        # Calculate accuracy (for demonstration purposes only, not for actual signature detection)
        # Replace this with a more appropriate accuracy calculation method for signature detection
        accuracy = 100 - num_faces

        # Print the number of detected faces and accuracy
        print("Number of detected faces:", num_faces)
        print("Accuracy:", accuracy)
        

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print("Error:", e)

if _name_ == "_main_":
    main()

# Rest of your code...