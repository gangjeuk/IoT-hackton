import math

import cv2
import numpy as np

#
# Codes from
# https://github.com/automaticdai/rpi-object-detection

CAMERA_DEVICE_ID = 0
IMAGE_WIDTH = 320
IMAGE_HEIGHT = 240
AT_LEAST = 0.0000000001

hsv_min = np.array((50, 80, 80))
hsv_max = np.array((120, 255, 255))

colors = []

def isset(v):
    try:
        type(eval(v))
    except:
        return 0
    else:
        return 1



def on_mouse_click(event, x, y, flags, frame):
    global colors

    if event == cv2.EVENT_LBUTTONUP:
        color_bgr = frame[y, x]
        color_rgb = tuple(reversed(color_bgr))

        print(color_rgb)

        color_hsv = rgb2hsv(color_rgb[0], color_rgb[1], color_rgb[2])
        print(color_hsv)

        colors.append(color_hsv)
        print(colors)

def hsv2rgb(h, s, v):
    h = float(h) * 2
    s = (float(s) / 255) + AT_LEAST
    v = (float(v) / 255) + AT_LEAST
    h60 = h / 60.0 + AT_LEAST
    h60f = math.floor(h60)
    hi = int(h60f) % 6
    f = h60 - h60f
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    r, g, b = 0, 0, 0
    if hi == 0:
        r, g, b = v, t, p
    elif hi == 1:
        r, g, b = q, v, p
    elif hi == 2:
        r, g, b = p, v, t
    elif hi == 3:
        r, g, b = p, q, v
    elif hi == 4:
        r, g, b = t, p, v
    elif hi == 5:
        r, g, b = v, p, q
    r, g, b = int(r * 255), int(g * 255), int(b * 255)
    return (r, g, b)


def rgb2hsv(r, g, b):
    r, g, b = r/255.0 , g/255.0, b/255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx-mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g-b)/df) + 360) % 360
    elif mx == g:
        h = (60 * ((b-r)/df) + 120) % 360
    elif mx == b:
        h = (60 * ((r-g)/df) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = df/mx
    v = mx

    h = int(h / 2)
    s = int(s * 255)
    v = int(v * 255)

    return (h, s, v)

def detect_by_color(cap :cv2.VideoCapture(CAMERA_DEVICE_ID)):
    try:

        cap.set(3, IMAGE_WIDTH)
        cap.set(4, IMAGE_HEIGHT)

        while True:
            _, frame = cap.read()
            frame = cv2.blur(frame, (3,3))

            hsv = cv2.cvtColor(frame, cv2, cv2.COLOR_BGR2HSV)
            cv2.setMouseCallback('frame', on_mouse_click, frame)


            if colors:
                minh = min(c[0] for c in colors)
                mins = min(c[1] for c in colors)
                minv = min(c[2] for c in colors)
                maxh = max(c[0] for c in colors)
                maxs = max(c[1] for c in colors)
                maxv = max(c[2] for c in colors)

                print("New HSV threshold: ", (minh, mins, minv), (maxh, maxs, maxv))
                hsv_min = np.array((minh, mins, minv))
                hsv_max = np.array((maxh, maxs, maxv))

            thresh = cv2.inRange(hsv, hsv_min, hsv_max)
            thresh2 = thresh.copy()


            (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
            print(major_ver, minor_ver, subminor_ver)

            # findContours() has different form for opencv2 and opencv3
            if major_ver == "2" or major_ver == "3":
                _, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            else:
                contours, hierarchy = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            # finding contour with maximum area and store it as best_cnt
            max_area = 0
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > max_area:
                    max_area = area
                    best_cnt = cnt
            # finding centroids of best_cnt and draw a circle there
            if isset('best_cnt'):
                M = cv2.moments(best_cnt)
                cx,cy = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
                cv2.circle(frame,(cx,cy),5,255,-1)
                print("Central pos: (%d, %d)" % (cx,cy))
            else:
                print("[Warning]Tag lost...")

            # Show the original and processed image
            #res = cv2.bitwise_and(frame, frame, mask=thresh2)
            cv2.imshow('frame', frame)
            cv2.imshow('thresh', thresh2)

            # if key pressed is 'Esc' then exit the loop
            if cv2.waitKey(33) == 27:
                break
    except Exception as e:
        print(e)
    finally:
        # Clean up and exit the program
        cv2.destroyAllWindows()
        cap.release()

def detect_by_feature(new_image, image_template):
    image1 = cv2.cvtColor(new_image, cv2.COLOR_BGR2GRAY)

    # Create ORB detector with 1000 keypoints with a scaling pyramid factor of 1.2
    orb = cv2.ORB_create(1000, 1.2)

    # Detect keypoints of original image
    (kp1, des1) = orb.detectAndCompute(image1, None)

    # Detect keypoints of rotated image
    (kp2, des2) = orb.detectAndCompute(image_template, None)

    # Create matcher
    # Note we're no longer using Flannbased matching
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # Do matching
    matches = bf.match(des1,des2)

    # Sort the matches based on distance.  Least distance
    # is better
    matches = sorted(matches, key=lambda val: val.distance)
    return len(matches)

def detect_by_shape(cap :cv2.VideoCapture(CAMERA_DEVICE_ID)):
    try:
        # set resolution to 320x240 to reduce latency
        cap.set(3, IMAGE_WIDTH)
        cap.set(4, IMAGE_HEIGHT)

        while True:
            # Read the frames frome a camera
            _, frame = cap.read()
            frame = cv2.blur(frame, (3, 3))

            # Or get it from a JPEG
            # frame = cv2.imread('frame0010.jpg', 1)

            # convert the image into gray color
            output = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # detect circles in the image
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 100)

            # ensure at least some circles were found
            if circles is not None:
                # convert the (x, y) coordinates and radius of the circles to integers
                circles = np.round(circles[0, :]).astype("int")
                # loop over the (x, y) coordinates and radius of the circles
                for (x, y, r) in circles:
                    # draw the circle in the output image, then draw a rectangle
                    # corresponding to the center of the circle
                    cv2.circle(output, (x, y), r, (0, 255, 0), 4)
                    cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)

            # show the output image
            cv2.imshow("frame", np.hstack([frame, output]))

            # if key pressed is 'Esc' then exit the loop
            if cv2.waitKey(33) == 27:
                break
    except Exception as e:
        print(e)
    finally:
        # Clean up and exit the program
        cv2.destroyAllWindows()
        cap.release()