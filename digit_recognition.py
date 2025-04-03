import cv2 as cv
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('C:\\Users\\jahan\\OneDrive\\Desktop\\PROG\\Projects\\ML Projects\\Hand Written Digit Recognition\\digit_recognition64.h5')

# Function to preprocess the image for prediction
def preprocess_image(img):
    # Resize to 28x28 pixels (assuming MNIST-like dataset)
    img = cv.resize(img, (28, 28))
    # Convert to grayscale
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #img = img / 255.0
    # Reshape to match the model input (1, 28, 28, 1)
    img = img.reshape(1, 28, 28, 1)
    return img

# Create a black canvas to draw digits
canvas = np.zeros((300, 300, 3), dtype='uint8')

move = False
ix, iy = -1, -1

def draw_digit(event, x, y, flags, param):
    global ix, iy, move
    radius = 8  # Circle radius for smooth drawing
    
    if event == cv.EVENT_LBUTTONDOWN:
        move = True
        ix, iy = x, y
        # Draw a circle at the starting point
        cv.circle(canvas, (ix, iy), radius, (255, 255, 255), -1, cv.LINE_AA)
    
    elif event == cv.EVENT_MOUSEMOVE:
        if move:
            # Draw circles at every move to ensure continuous strokes
            cv.line(canvas, (ix, iy), (x, y), (255, 255, 255), radius*2, cv.LINE_AA)
            ix, iy = x, y
    
    elif event == cv.EVENT_LBUTTONUP:
        move = False
        # Ensure the last point is connected
        cv.line(canvas, (ix, iy), (x, y), (255, 255, 255), radius*2, cv.LINE_AA)

# Set up the window and bind the mouse callback function
cv.namedWindow('Digit Canvas')
cv.setMouseCallback('Digit Canvas', draw_digit)

while True:
    # Display the canvas
    cv.imshow('Digit Canvas', canvas)
    
    # Capture user input for prediction, clearing, or exiting
    key = cv.waitKey(1) & 0xFF

    if key == ord('p'):
        # Show the drawn image before preprocessing
        cv.imshow("Original Drawn Image", canvas)
        
        # Preprocess the canvas image for prediction
        processed_img = preprocess_image(canvas)

        # Show the processed image (28x28) for verification
        processed_display_img = processed_img.reshape(28, 28)
        # Resize the 28x28 image to 280x280 for better visibility
        enlarged_img = cv.resize(processed_display_img, (280, 280), interpolation=cv.INTER_NEAREST)
        cv.imshow("Processed Image (Enlarged)", enlarged_img)

        # Predict the digit
        prediction = model.predict(processed_img)
        predicted_digit = np.argmax(prediction)
        print(f'Predicted Digit: {predicted_digit}')

        # Optionally, clear the canvas after each prediction
        canvas = np.zeros((300, 300, 3), dtype='uint8')
    
    elif key == ord('c'):
        # Clear the canvas
        canvas = np.zeros((300, 300, 3), dtype='uint8')

    elif key == ord('q'):
        # Quit the application
        break

# Close all windows
cv.destroyAllWindows()