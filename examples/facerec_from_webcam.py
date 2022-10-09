import face_recognition
import cv2
import numpy as np
import time
# This is a super simple (but slow) example of running face recognition on live video from your webcam.
# There's a second example that's a little more complicated but runs faster.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Load a sample picture and learn how to recognize it.
obama_image = face_recognition.load_image_file("obama.jpg")
obama_face_encoding = face_recognition.face_encodings(obama_image)[0]

# Load a second sample picture and learn how to recognize it.
biden_image = face_recognition.load_image_file("biden.jpg")
biden_face_encoding = face_recognition.face_encodings(biden_image)[0]

diaofeng_image = face_recognition.load_image_file("diaofeng.jpeg")
diaofeng_face_encoding = face_recognition.face_encodings(diaofeng_image)[0]
print("Face Key Point Matrix : ")
print(diaofeng_face_encoding)
print("Face Key Point Num : ")
print(len(diaofeng_face_encoding))

chuanchuan_image = face_recognition.load_image_file("chuanchuan.jpeg")
chuanchuan_face_encoding = face_recognition.face_encodings(chuanchuan_image)[0]

# Create arrays of known face encodings and their names
known_face_encodings = [
    obama_face_encoding,
    biden_face_encoding,
    diaofeng_face_encoding,
    chuanchuan_face_encoding
]
known_face_names = [
    "Barack Obama",
    "Joe Biden",
    "diaofeng",
    "chuanchuan"

]


while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()
    # 记录该帧开始处理的时间
    start_time = time.time()

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces and face enqcodings in the frame of video
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Loop through each face in this frame of video
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # See if the face is a match for the known face(s)
        # 数据库中的人脸特征与当前帧中的人脸特征进行比对
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"

        # If a match was found in known_face_encodings, just use the first one.
        # if True in matches:
        #     first_match_index = matches.index(True)
        #     name = known_face_names[first_match_index]

        # Or instead, use the known face with the smallest distance to the new face
        # 计算欧式距离
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        print(len(face_distances))
        print(face_distances)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # 记录该帧处理完毕的时间
    end_time = time.time()
    # 计算每秒处理图像帧数FPS
    FPS = 1/(end_time - start_time)
    # print("FPS: ", FPS)
    frame = cv2.putText(frame, 'FPS  '+str(int(FPS)), (25, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.25, (255, 0, 255), 2)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
