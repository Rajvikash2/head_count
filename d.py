import cv2

# Load the Haar Cascade for face detection
head_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Read the image from file
image_path = "cr1.jpg"  
image = cv2.imread(image_path)


if image is None:
    print("Error: Unable to load image.")
else:
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    gray_image = cv2.equalizeHist(gray_image)

   
    heads = head_cascade.detectMultiScale(
        gray_image,
        scaleFactor=1.1,  # Try a value between 1.05 to 1.3 for better accuracy
        minNeighbors=5,  
        minSize=(30, 30), 
        maxSize=(300, 300),
    )

    # Draw rectangles around detected heads
    for (x, y, w, h) in heads:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the number of heads detected
    print(f"Number of humans detected: {len(heads)}")

    # Display the image with detections
    cv2.imshow("Image Viewer - Head Detection", image)

    # Wait for a key press indefinitely or for a specific time in milliseconds
    cv2.waitKey(0) 
    cv2.destroyAllWindows()
