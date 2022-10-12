import cv2
from models import models

cap = cv2.VideoCapture(0)

# Load our image template, this is our reference image
image_template = cv2.imread('../src/simple.png', 0)
src_image = cv2.imread('../src/simple.png', 0)

while True:
    # Get webcam images
    ret, frame = cap.read()
    ret, frame = (False, src_image)
    # Get height and width of webcam frame
    height, width = frame.shape[:2]

    # Define ROI Box Dimensions (Note some of these things should be outside the loop)
    top_left_x = int(width / 3)
    top_left_y = int((height / 2) + (height / 4))
    bottom_right_x = int((width / 3) * 2)
    bottom_right_y = int((height / 2) - (height / 4))

    # Draw rectangular window for our region of interest
    cv2.rectangle(frame, (top_left_x,top_left_y), (bottom_right_x,bottom_right_y), 255, 3)

    # Crop window of observation we defined above
    cropped = frame[bottom_right_y:top_left_y , top_left_x:bottom_right_x]

    # Flip frame orientation horizontally
    frame = cv2.flip(frame,1)

    # Get number of ORB matches
    matches = models.detect_by_feature(src_image, image_template)

    # Display status string showing the current no. of matches
    output_string = "# of Matches = " + str(matches)
    cv2.putText(frame, output_string, (50,450), cv2.FONT_HERSHEY_COMPLEX, 1, (250,0,0), 2)

    # Our threshold to indicate object deteciton
    # For new images or lightening conditions you may need to experiment a bit
    # Note: The ORB detector to get the top 1000 matches, 350 is essentially a min 35% match
    threshold = 200

    # If matches exceed our threshold then object has been detected
    if matches > threshold:
        cv2.rectangle(frame, (top_left_x,top_left_y), (bottom_right_x,bottom_right_y), (0,255,0), 3)
        cv2.putText(frame,'Object Found',(50,50), cv2.FONT_HERSHEY_COMPLEX, 2 ,(0,255,0), 2)

    cv2.imshow('Object Detector using ORB', frame)
    if cv2.waitKey(1) == 13: #13 is the Enter Key
        break

cap.release()
cv2.destroyAllWindows()