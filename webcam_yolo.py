# import os
# os.environ["YOLO_OFFLINE"] = "True"

# import cv2
# from ultralytics import YOLO
# import time

# model = YOLO(r"D:\5th Semester files\ML\ML Lab Manuals\ML_AR_Object_Classifier\inference\best.pt")

# cap = cv2.VideoCapture(0)

# prev_time = 0

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     results = model(frame)

#     current_time = time.time()
#     fps = 1 / (current_time - prev_time)
#     prev_time = current_time

#     annotated = results[0].plot()
#     cv2.putText(annotated, f"FPS: {int(fps)}", (20,40),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

#     cv2.imshow("YOLO AR Object Detection", annotated)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()




import os
os.environ["YOLO_OFFLINE"] = "True"

import cv2
from ultralytics import YOLO
import time

# 🔹 Load trained model
# model = YOLO(r"D:\5th Semester files\ML\ML Lab Manuals\ML_AR_Object_Classifier\inference\best.pt")

model = YOLO(r"D:\5th Semester files\ML\ML Lab Manuals\ML_AR_Object_Classifier\inference\yolov8n (1).pt")




# 🔹 Open webcam
cap = cv2.VideoCapture(0)

# 🔹 Set camera resolution (if supported)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# 🔹 Create resizable window
cv2.namedWindow("YOLO AR Object Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("YOLO AR Object Detection", 1280, 720)

prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Webcam not accessible")
        break

    # 🔹 YOLO inference
    results = model(frame, conf=0.5)

    # 🔹 FPS calculation
    current_time = time.time()
    fps = 1 / (current_time - prev_time) if prev_time != 0 else 0
    prev_time = current_time

    # 🔹 Draw detections
    annotated = results[0].plot()

    # 🔹 Show FPS
    cv2.putText(
        annotated,
        f"FPS: {int(fps)}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # 🔹 Display output
    cv2.imshow("YOLO AR Object Detection", annotated)

    # 🔹 Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 🔹 Release resources
cap.release()
cv2.destroyAllWindows()
