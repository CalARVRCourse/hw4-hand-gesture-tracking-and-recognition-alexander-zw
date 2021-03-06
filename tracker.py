"""
Includes demos of gesture tracking and recognition as required in video.
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

def add_text(img, text, location, scale=1):
    cv2.putText(img, text, location, cv2.FONT_HERSHEY_COMPLEX, scale, [max_color,0,max_color], thickness=3)

def add_text_top_left(img, text, scale=1):
    add_text(img, text, (50, 50), scale=scale)

def add_text_bottom_left(img, text, scale=1):
    add_text(img, text, (50, 670), scale=scale)

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

def threshold_binarize(frame, invert=False):
    """ Returns the threshold value and the binarized image. """
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    thresh_type = (cv2.THRESH_BINARY_INV if invert else 0) + cv2.THRESH_OTSU
    return cv2.threshold(gray, 0, max_color, thresh_type)

def get_connected_components(frame):
    ret, markers, stats, centroids = cv2.connectedComponentsWithStats(frame, ltype=cv2.CV_16U)
    markers = np.array(markers, dtype=np.uint8)
    label_hue = np.uint8(179 * markers / np.max(markers))
    blank_ch = max_color * np.ones_like(label_hue)
    labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])
    labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)
    labeled_img[label_hue==0] = 0
    return ret, stats, labeled_img

def detect_hand_ellipse_hole(binary_frame, annotate=False):
    """
    Returns the ellipse center, minor and major axis lengths, angle, and annotated frame.
    Tries to detect a hole in the hand gesture, and if detected returns the parameters of
    an ellipse matching the contour. If no ellipse detected, returns 0 for all parameters.
    If annotate, display ellipse in a separate window.
    """
    ret, stats, labeled_img = get_connected_components(binary_frame)
    c_x, c_y, MA, ma, angle, frame = 0, 0, 0, 0, 0, binary_frame
    if ret <= 2:
        return c_x, c_y, MA, ma, angle, frame
    if annotate:
        frame = cv2.cvtColor(binary_frame, cv2.COLOR_GRAY2RGB) # Convert back to color.
    try:
        stats_sorted_by_area = stats[np.argsort(stats[:, 4])]
        roi = stats_sorted_by_area[-3][0:4]
        x, y, w, h = roi
        subimg = labeled_img[y:y+h, x:x+w]
        subimg = cv2.cvtColor(subimg, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(subimg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        max_contour = max(contours, key=len)
        (c_x,c_y), (MA,ma), angle = cv2.fitEllipse(max_contour)
        c_x, c_y, MA, ma, angle = round(x + c_x), round(y + c_y), round(MA, 1), round(ma, 1), round(angle, 1)

        if annotate:
            if len(max_contour) >= 5:
                ellipseParam = cv2.fitEllipse(max_contour)
                subimg = cv2.cvtColor(subimg, cv2.COLOR_GRAY2RGB)
                subimg = cv2.ellipse(subimg, ellipseParam, (0,max_color,0), 2) # Add green ellipse.
            cv2.circle(frame, (c_x, c_y), 5, [max_color,0,max_color], -1) # Purple center.
            subimg = cv2.resize(subimg, (0,0), fx=3, fy=3)
            add_text_top_left(frame, f"({c_x}, {c_y}); axes {MA}, {ma}; angle {angle}")
            add_text_bottom_left(frame, f"area {round(MA * ma)}; axis ratio {round(ma / MA, 3)}")
            imshow_smaller("ROI 2", subimg)
    except:
        if annotate:
            add_text_top_left(frame, "No hand found")
    return c_x, c_y, MA, ma, angle, frame

def detect_fingers(binary_frame, annotate=False):
    """
    Returns number of fingers, area of hand, center of hand, and the annotated image.
    If cannot detect hand, returns 1 for number of fingers and 0 for other parameters.
    """
    contours, _ = cv2.findContours(binary_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    finger_count, area, cX, cY, frame = 1, 0, 0, 0, binary_frame
    if len(contours) <= 1:
        return finger_count, area, cX, cY, frame
    largest_contour = max(contours, key=cv2.contourArea)
    hull = cv2.convexHull(largest_contour, returnPoints = False)
    frame = binary_frame
    if annotate:
        frame = cv2.cvtColor(binary_frame, cv2.COLOR_GRAY2RGB) # Convert back to color.
        cv2.fillPoly(frame, pts=[largest_contour], color=[255,200,100]) # Fill largest contour with light blue.
    # Unless there are no fingers, the actual number is + 1 since we count valleys not fingers.
    for cnt in contours[:1]:
        try:
            defects = cv2.convexityDefects(cnt, hull)
            if defects is not None:
                for i in range(defects.shape[0]):
                    s, e, f, d = defects[i,0]
                    start = tuple(cnt[s][0])
                    end = tuple(cnt[e][0])
                    far = tuple(cnt[f][0])

                    if annotate:
                        cv2.line(frame, start, end, [0,max_color,0], 2) # Green convex hull.
                        cv2.circle(frame, far, 5, [0,200,max_color], -1) # Orange points.

                    if is_finger(start, end, far):
                        finger_count += 1
                        if annotate:
                            cv2.circle(frame, far, 4, [0,0,max_color], -1) # Red finger valley.

        except cv2.error:
            pass
    M = cv2.moments(largest_contour)
    area = M["m00"]
    cX = round(M["m10"] / area)
    cY = round(M["m01"] / area)
    if annotate:
        cv2.circle(frame, (cX, cY), 5, [max_color,0,max_color], -1) # Purple center.
        add_text_top_left(frame, f"({cX}, {cY}); {finger_count} fingers; area {area}")
    return finger_count, area, cX, cY, frame

def is_finger(start, end, far):
    c_squared = (end[0] - start[0])**2 + (end[1] - start[1])**2
    a_squared = (far[0] - start[0])**2 + (far[1] - start[1])**2
    b_squared = (end[0] - far[0])**2 + (end[1] - far[1])**2
    angle = np.arccos((a_squared + b_squared  - c_squared ) / (2 * np.sqrt(a_squared * b_squared)))
    return angle <= np.pi / 3


window_name = "Hand Gesture Tracking"
max_color = 255
DETECT_HOLE = True # Used to switch between detecting hole and fingers.

if __name__ == "__main__":
    cam = cv2.VideoCapture(0)
    cv2.namedWindow(window_name)

    print("Starting gesture detection")
    while True:
        ret, frame = cam.read()
        if not ret:
            print("Could not find ret, exiting")
            break

        skin_mask = hsv_and_ycrcb_mask(frame)
        frame = blur_noise(frame, skin_mask)

        if DETECT_HOLE:
            _, binary_frame = threshold_binarize(frame, invert=True)
            frame = detect_hand_ellipse_hole(binary_frame, annotate=True)[-1]
        else:
            _, binary_frame = threshold_binarize(frame, invert=False)
            frame = detect_fingers(binary_frame, annotate=True)[-1]

        imshow_smaller(window_name, frame)

        k = cv2.waitKey(1) # k is the key pressed.
        if k == 27 or k == 113:  # 27, 113 are ascii for escape and q respectively.
            print("Received user input, exiting")
            cv2.destroyAllWindows()
            cam.release()
            break

