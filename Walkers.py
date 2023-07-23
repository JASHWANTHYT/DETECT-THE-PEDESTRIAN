import cv2

# Load the pre-trained Haar Cascade classifier for human body detection
body_classifier = cv2.CascadeClassifier('D:\whjr new\PRO-106-ProjectTemplate-main\haarcascade_fullbody.xml')

# Open a video capture object
cap = cv2.VideoCapture('D:\whjr new\PRO-106-ProjectTemplate-main\walking.avi')

while cap.isOpened():
    # Read a frame from the video
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect human bodies in the frame
    bodies = body_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Draw rectangles around the detected bodies
    for (x, y, w, h) in bodies:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the frame with rectangles
    cv2.imshow('Human Body Detection', frame)

    # Check if the user pressed the 'q' key to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close the OpenCV windows
cap.release()
cv2.destroyAllWindows()
