import mediapipe as mp
import cv2 as cv
import math
import pyfirmata

arduino = pyfirmata.Arduino("COM5")
previous_state = 0
orginal = previous_state
LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
m = 0
mp_mesh = mp.solutions.face_mesh
mp_styles = mp.solutions.drawing_styles
mp_draw = mp.solutions.drawing_utils

video = cv.VideoCapture(0)
face = mp_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)


def distance(point, point1):
    x1, y1 = point
    x2, y2 = point1
    dist = math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
    return dist

def blink_ratio(landmarks, right, left):
    rh_right = landmarks[right[0]]
    rh_left = landmarks[right[8]]

    rv_top = landmarks[right[12]]
    rv_bottom = landmarks[right[4]]

    lh_right = landmarks[left[0]]
    lh_left = landmarks[left[8]]

    lh_top = landmarks[left[12]]
    lh_bottom = landmarks[left[4]]

    rh_eye = distance(rh_left, rh_right)
    rv_eye = distance(rv_top, rv_bottom)

    lh_eye = distance(lh_left, lh_right)
    lv_eye = distance(lh_top, lh_bottom)
    try:
        re_ratio = rh_eye / rv_eye
        le_ratio = lh_eye / lv_eye
        blink_rati = (re_ratio + le_ratio) / 2
        return blink_rati
    except ZeroDivisionError:
        return 1.0

while True:
    status, image = video.read()
    img_height, img_width = image.shape[:2]
    if not status:
        break
    else:
        image.flags.writeable = False
        black = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image.flags.writeable = True
        res = face.process(black)
        face_landmarks = res.multi_face_landmarks
        if face_landmarks:
            for face_landmark in res.multi_face_landmarks:
                mp_draw.draw_landmarks(image=image, landmark_list=face_landmark,
                                       connections=mp_mesh.FACEMESH_TESSELATION,
                                       landmark_drawing_spec=None,
                                       connection_drawing_spec=mp_styles.get_default_face_mesh_tesselation_style())
                mp_draw.draw_landmarks(image=image, landmark_list=face_landmark, connections=mp_mesh.FACEMESH_CONTOURS,
                                       landmark_drawing_spec=None,
                                       connection_drawing_spec=mp_styles.get_default_face_mesh_contours_style())
                mp_draw.draw_landmarks(image=image, landmark_list=face_landmark, connections=mp_mesh.FACEMESH_IRISES,
                                       landmark_drawing_spec=None,
                                       connection_drawing_spec=mp_styles.get_default_face_mesh_iris_connections_style())
            mesh_cord = [(int(point.x * img_width), int(point.y * img_height)) for point in face_landmarks[0].landmark]
            ratio = blink_ratio(mesh_cord, RIGHT_EYE, LEFT_EYE)
            if ratio > 4.0:
                m += 1
                arduino.digital[8].write(0)

                print("closed")
                print(m)
            else:
                arduino.digital[8].write(1)

    cv.imshow('frame', image)
    key = cv.waitKey(2)
    if key == ord('q') or key == ord('Q'):
        break
cv.destroyAllWindows()
video.release()
