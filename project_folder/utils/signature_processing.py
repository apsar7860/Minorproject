import os
import cv2
import numpy as np

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

def calculate_accuracy(original_signature, processed_signature):
    # Compute the mean squared error between the original and processed signatures
    mse = np.mean((original_signature - processed_signature) ** 2)

    # Compute the accuracy (lower MSE indicates better accuracy)
    accuracy = 100 - mse

    return accuracy

def main():
    # Prompt the user to enter the signature image file name or path
    signature_path = input("Enter the signature image file name or path: ")

    try:
        # Load the original signature image
        original_signature = cv2.imread(signature_path, cv2.IMREAD_GRAYSCALE)

        # Process the signature image
        processed_signature = process_signature(signature_path)

        # Calculate accuracy
        accuracy = calculate_accuracy(original_signature, processed_signature)

        # Print the accuracy
        print("Accuracy:", accuracy)

        # Assuming 'processed_signature' is the processed signature you obtained
        output_folder = r'C:\Users\Nehat\Desktop\project_folder\static\images'

        output_path = os.path.join(output_folder, 'processed_signature.png')
        #print("Output path:", output_path)

        #if os.path.exists(output_path):
            #print("The file exists at:", output_path)
        #else:
            #print("The file does not exist at:", output_path)

        # Show the original and processed signatures (uncomment these lines if you want to display the images)
        cv2.imshow("Original Signature", original_signature)
        cv2.imshow("Processed Signature", processed_signature)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print("Error:", e)

if _name_ == "_main_":
    main()