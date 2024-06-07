import os
from datetime import date, datetime

import cv2
import joblib
import mysql.connector
import numpy as np
from flask import Flask, request, render_template
from sklearn.neighbors import KNeighborsClassifier

app = Flask(__name__)

nimgs = 10

imgBackground = cv2.imread("background.png")

datetoday = date.today().strftime("%m_%d_%y")
datetoday2 = date.today().strftime("%d-%B-%Y")

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# MySQL Connection Configuration
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': '',
    'database': 'biometricattendace'
}

# Establish MySQL Connection
conn = mysql.connector.connect(**db_config)
cursor = conn.cursor()


# Function to add attendance to MySQL database
def add_attendance_to_mysql(name, facerecognition_id):
    try:
        current_time = datetime.now().strftime("%H:%M:%S")
        query = "INSERT INTO Attendance (Name, facerecognition_id, Time) VALUES (%s, %s, %s)"
        data = (name, facerecognition_id, current_time)
        cursor.execute(query, data)
        conn.commit()
    except mysql.connector.Error as error:
        print("Failed to insert record into Attendance table:", error)


# Replace the add_attendance function in your code with the following:
def add_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    add_attendance_to_mysql(username, userid)


if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,facerecognition_id,Time')


def totalreg():
    return len(os.listdir('static/faces'))


def extract_faces(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
        return face_points
    except:
        return []


def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)


def train_model():
    faces = []
    labels = []
    userlist = os.listdir('static/faces')
    for user in userlist:
        for imgname in os.listdir(f'static/faces/{user}'):
            img = cv2.imread(f'static/faces/{user}/{imgname}')
            resized_face = cv2.resize(img, (50, 50))
            faces.append(resized_face.ravel())
            labels.append(user)
    faces = np.array(faces)
    knn = KNeighborsClassifier(n_neighbors=2)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')


def extract_attendance():
    try:
        cursor.execute("SELECT Name, facerecognition_id, Time FROM Attendance")
        records = cursor.fetchall()
        names = [record[0] for record in records]
        facerecognition_ids = [record[1] for record in records]
        times = [record[2] for record in records]
        l = len(records)
        return names, facerecognition_ids, times, l
    except mysql.connector.Error as error:
        print("Failed to fetch attendance records:", error)
        return [], [], [], 0


@app.route('/')
def home():
    names, facerecognition_ids, times, l = extract_attendance()
    return render_template('home.html', names=names, facerecognition_ids=facerecognition_ids, times=times, l=l,
                           totalreg=totalreg(), datetoday2=datetoday2)


@app.route('/start', methods=['GET'])
def start():
    names, facerecognition_ids, times, l = extract_attendance()

    if 'face_recognition_model.pkl' not in os.listdir('static'):
        return render_template('home.html', names=names, facerecognition_ids=facerecognition_ids, times=times, l=l,
                               totalreg=totalreg(), datetoday2=datetoday2,
                               mess='There is no trained model in the static folder. Please add a new face to continue.')

    ret = True
    cap = cv2.VideoCapture(1,cv2.CAP_DSHOW)
    while ret:
        ret, frame = cap.read()
        if len(extract_faces(frame)) > 0:
            (x, y, w, h) = extract_faces(frame)[0]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (86, 32, 251), 1)
            cv2.rectangle(frame, (x, y), (x + w, y - 40), (86, 32, 251), -1)
            face = cv2.resize(frame[y:y + h, x:x + w], (50, 50))
            identified_person = identify_face(face.reshape(1, -1))[0]
            add_attendance(identified_person)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 2)
            cv2.rectangle(frame, (x, y - 40), (x + w, y), (50, 50, 255), -1)
            cv2.putText(frame, f'{identified_person}', (x, y - 15), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 255), 1)
        imgBackground[162:162 + 480, 55:55 + 640] = frame
        cv2.imshow('Attendance', imgBackground)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    names, facerecognition_ids, times, l = extract_attendance()
    return render_template('home.html', names=names, facerecognition_ids=facerecognition_ids, times=times, l=l,
                           totalreg=totalreg(), datetoday2=datetoday2)


@app.route('/add', methods=['GET', 'POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    userimagefolder = 'static/faces/' + newusername + '_' + str(newuserid)
    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)
    i = 0  # Counter for captured images
    cap = cv2.VideoCapture(1,cv2.CAP_DSHOW)
    while 1:
        _, frame = cap.read()
        faces = extract_faces(frame)
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
            cv2.putText(frame, f'Images Captured: {i}/{2}', (30, 30),  # Capture only 2 images
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20), 2, cv2.LINE_AA)
            name = newusername + '_' + str(i) + '.jpg'
            cv2.imwrite(userimagefolder + '/' + name, frame[y:y + h, x:x + w])
            i += 1
        if i == 5:  # Capture only 2 images
            break
        cv2.imshow('Adding new User', frame)
        if cv2.waitKey(1) == 27:
            break
    cap.release()
    cv2.destroyAllWindows()
    print('Training Model')
    train_model()
    names, facerecognition_ids, times, l = extract_attendance()
    return render_template('home.html', names=names, facerecognition_ids=facerecognition_ids, times=times, l=l,
                           totalreg=totalreg(), datetoday2=datetoday2)


if __name__ == '__main__':
    app.run(debug=True)
