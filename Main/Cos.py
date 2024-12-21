import time

import cv2

cam_port = 0
cam = cv2.VideoCapture(cam_port)
if not cam.isOpened():
    print("Error: Could not access the camera")
else:
    time.sleep(2)
    result, image = cam.read()

    if result:
        cv2.imshow("Captured Image", image)
        cv2.imwrite("captured_image.png", image)
        print("Captured Image")

        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No image detected. Please try again.")
cam.release()