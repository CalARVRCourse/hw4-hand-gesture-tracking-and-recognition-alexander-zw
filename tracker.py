"""
Main code to run gesture tracking and recognition.
"""
import cv2
import numpy as np

def resize(img, scale=0.5):
    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    dim = (width, height)
    # resize image
    return cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

def imshow_smaller(title, img, scale=0.75):
    img_small = resize(img, scale=scale)
    cv2.imshow(title, img_small)

def hsv_and_ycrcb_mask(frame):
    lower_HSV = np.array([0, 40, 0], dtype = "uint8")
    upper_HSV = np.array([25, 255, 255], dtype = "uint8")
    
    converted_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    skin_mask_HSV = cv2.inRange(converted_HSV, lower_HSV, upper_HSV)

    lower_YCrCb = np.array([0, 138, 67], dtype = "uint8")
    upper_YCrCb = np.array((255, 173, 133), dtype = "uint8")
        
    converted_YCrCb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
    skin_mask_YCrCb = cv2.inRange(converted_YCrCb, lower_YCrCb, upper_YCrCb)
    
    return cv2.add(skin_mask_HSV, skin_mask_YCrCb)

def blur_noise(frame, skin_mask):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    skin_mask = cv2.erode(skin_mask, kernel, iterations = 2)
    skin_mask = cv2.dilate(skin_mask, kernel, iterations = 2)
    
    # Blur the mask to help remove noise, then apply the mask to the frame.
    skin_mask = cv2.GaussianBlur(skin_mask, (3, 3), 0)
    return cv2.bitwise_and(frame, frame, mask=skin_mask)

def threshold_binarize(frame):
    """ Returns the threshold value and the binarized image. """
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    return cv2.threshold(gray, 0, max_color, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

def get_connected_components(frame):
    ret, markers, stats, centroids = cv2.connectedComponentsWithStats(frame, ltype=cv2.CV_16U)
    markers = np.array(markers, dtype=np.uint8)
    label_hue = np.uint8(179 * markers / np.max(markers))
    blank_ch = max_color * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
    labeled_img[label_hue==0] = 0
    return ret, stats, labeled_img

def find_and_show_hand_countour(frame):
    """
    Tries to detect a contour in the hand gesture, and if detected displays an ellipse
    matching the contour in a separate window.
    """
    ret, stats, labeled_img = get_connected_components(frame)
    if ret > 2:
        try:
            stats_sorted_by_area = stats[np.argsort(stats[:, 4])]
            roi = stats_sorted_by_area[-3][0:4]
            x, y, w, h = roi
            subimg = labeled_img[y:y+h, x:x+w]
            subimg = cv2.cvtColor(subimg, cv2.COLOR_BGR2GRAY)
            contours, _ = cv2.findContours(subimg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            max_contour = max(contours, key=len)
            if len(max_contour) >= 5:
                ellipseParam = cv2.fitEllipse(max_contour)
                subimg = cv2.cvtColor(subimg, cv2.COLOR_GRAY2RGB)
                subimg = cv2.ellipse(subimg, ellipseParam, (0,max_color,0), 2) # Add green ellipse.
            
            subimg = cv2.resize(subimg, (0,0), fx=3, fy=3)
            (x,y), (MA,ma), angle = cv2.fitEllipse(max_contour)
            print(f"Ellipse detected: center {x}, {y}; axis lengths {MA}, {ma}; angle {angle}")
            imshow_smaller("ROI 2", subimg)
        except:
            print("No hand found")

window_name = "Hand Gesture Tracking"
# lower_hsv_trackbar = "lower_hsv"
# upper_hsv_trackbar = "upper_hsv"
# lower_ycrcb_trackbar = "lower_ycrcb"
# upper_ycrcb_trackbar = "upper_ycrcb"
# max_trackbar_val = 200
max_color = 255

if __name__ == "__main__":
    cam = cv2.VideoCapture(0)
    cv2.namedWindow(window_name)
    # Create a Trackbar to choose a value for a parameter.
    # cv2.createTrackbar(lower_hsv_trackbar, window_name, 0, max_trackbar_val, lambda x: None)
    # cv2.createTrackbar(upper_hsv_trackbar, window_name, 0, max_trackbar_val, lambda x: None)
    # cv2.createTrackbar(lower_ycrcb_trackbar, window_name, 0, max_trackbar_val, lambda x: None)
    # cv2.createTrackbar(upper_ycrcb_trackbar, window_name, 0, max_trackbar_val, lambda x: None)

    print("Starting gesture detection")
    while True:
        ret, frame = cam.read()
        if not ret:
            print("Could not find ret, exiting")
            break

        # hsv1 = cv2.getTrackbarPos(lower_hsv_trackbar, window_name)
        # hsv2 = cv2.getTrackbarPos(upper_hsv_trackbar, window_name)
        # ycrcb1 = cv2.getTrackbarPos(lower_ycrcb_trackbar, window_name)
        # ycrcb2 = cv2.getTrackbarPos(upper_ycrcb_trackbar, window_name)

        skin_mask = hsv_and_ycrcb_mask(frame)
        frame = blur_noise(frame, skin_mask)

        ret, frame = threshold_binarize(frame)
        find_and_show_hand_countour(frame)

        imshow_smaller(window_name, frame)
        k = cv2.waitKey(1) # k is the key pressed.
        if k == 27 or k == 113:  # 27, 113 are ascii for escape and q respectively.
            print("Received user input, exiting")
            cv2.destroyAllWindows()
            cam.release()
            break

