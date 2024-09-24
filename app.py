import cv2
import os
from flask import Flask, request, render_template, send_file
from datetime import date
from datetime import datetime
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import joblib

# Defining Flask App
app = Flask(__name__)

# Number of images to take for each user
nimgs = 10

# Saving Date today in 2 different formats
datetoday = date.today().strftime("%m_%d_%y")

# Path to the CSV and Excel files
def get_csv_path():
    return os.path.join('Attendance', f'Attendance-{datetoday}.csv')

def get_excel_path():
    return os.path.join('Attendance', f'Attendance-{datetoday}.xlsx')



# Initializing VideoCapture object to access WebCam
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


# If these directories don't exist, create them
if not os.path.isdir('Attendance'):
    os.makedirs('Attendance')
if not os.path.isdir('static'):
    os.makedirs('static')
if not os.path.isdir('static/faces'):
    os.makedirs('static/faces')
if f'Attendance-{datetoday}.csv' not in os.listdir('Attendance'):
    with open(f'Attendance/Attendance-{datetoday}.csv', 'w') as f:
        f.write('Name,Roll,Time')

# Convert CSV to Excel function
def csv_to_excel(csv_file, excel_file):
    try:
        print(f"Converting CSV: {csv_file} to Excel: {excel_file}")
        df = pd.read_csv(csv_file)
        df.to_excel(excel_file, index=False, engine='openpyxl')
        print(f"Successfully converted {csv_file} to {excel_file}")
    except Exception as e:
        print(f"Failed to convert CSV to Excel: {e}")

# get a number of total registered users
def totalreg():
    return len(os.listdir('static/faces'))


# extract the face from an image
def extract_faces(img):
    try:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        face_points = face_detector.detectMultiScale(gray, 1.2, 5, minSize=(20, 20))
        return face_points if len(face_points) > 0 else None  # Return None if no faces are found
    except Exception as e:
        print(f"Error extracting faces: {e}")
        return None


# Identify face using ML model
def identify_face(facearray):
    model = joblib.load('static/face_recognition_model.pkl')
    return model.predict(facearray)


# A function which trains the model on all the faces available in faces folder
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
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(faces, labels)
    joblib.dump(knn, 'static/face_recognition_model.pkl')


# Extract info from today's attendance file in attendance folder
def extract_attendance():
    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    names = df['Name']
    rolls = df['Roll']
    times = df['Time']
    l = len(df)
    return names, rolls, times, l


# Add Attendance of a specific user
def add_attendance(name):
    username = name.split('_')[0]
    userid = name.split('_')[1]
    current_time = datetime.now().strftime("%H:%M:%S")

    df = pd.read_csv(f'Attendance/Attendance-{datetoday}.csv')
    if int(userid) not in list(df['Roll']):
        with open(f'Attendance/Attendance-{datetoday}.csv', 'a') as f:
            f.write(f'\n{username},{userid},{current_time}')



################## ROUTING FUNCTIONS #######################
####### for Face Recognition based Attendance System #######

# Our main page
@app.route('/')
def home():
    names, rolls, times, l = extract_attendance()
    csv_file = get_csv_path()

    # Ensure CSV file exists (simulate attendance file creation)
    if not os.path.exists(csv_file):
        with open(csv_file, 'w') as f:
            f.write('Name,Roll,Time\n')
            f.write('John Doe,101,09:00:00\n')

    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg())


# Our main Face Recognition functionality.
# This function will run when we click on Take Attendance Button.
@app.route('/start', methods=['GET'])
def start():
    names, rolls, times, l = extract_attendance()

    if not os.path.exists('static/face_recognition_model.pkl'):
        return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday, mess='Face recognition model not found. Please add a new user.')

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday, mess='Could not open webcam.')

    ret = True
    while ret:
        ret, frame = cap.read()
        if not ret:
            break  # Break the loop if frame not captured

        face_points = extract_faces(frame)
        if face_points is not None and len(face_points) > 0:  # Ensure faces are detected
            for (x, y, w, h) in face_points:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (86, 32, 251), 1)
                cv2.rectangle(frame, (x, y), (x + w, y - 40), (86, 32, 251), -1)
                face = cv2.resize(frame[y:y + h, x:x + w], (50, 50))

                try:
                    identified_person = identify_face(face.reshape(1, -1))[0]
                    add_attendance(identified_person)
                    cv2.putText(frame, f'{identified_person}', (x + 5, y - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                except Exception as e:
                    print(f"Error identifying face: {e}")

        else:
            print("No faces detected")

        cv2.imshow('Attendance', frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    names, rolls, times, l = extract_attendance()
    return render_template('home.html', names=names, rolls=rolls, times=times, l=l, totalreg=totalreg(), datetoday2=datetoday)

# A function to add a new user.
# This function will run when we add a new user.
@app.route('/add', methods=['GET', 'POST'])
def add():
    newusername = request.form['newusername']
    newuserid = request.form['newuserid']
    userimagefolder = f'static/faces/{newusername}_{newuserid}'

    if not os.path.isdir(userimagefolder):
        os.makedirs(userimagefolder)

    cap = cv2.VideoCapture(0)  # Open camera

    if not cap.isOpened():  # Check if the camera opened successfully
        return render_template('home.html',
                               mess="Could not open webcam. Please ensure it is connected and not being used by another application.")

    i, j = 0, 0
    while True:
        ret, frame = cap.read()
        if not ret:
            return render_template('home.html', mess="Failed to capture image from camera.")

        faces = extract_faces(frame)

        if faces is not None:
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 20), 2)
                cv2.putText(frame, f'Images Captured: {i}/{nimgs}', (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 20),
                            2, cv2.LINE_AA)

                if j % 5 == 0:
                    name = f'{newusername}_{i}.jpg'
                    cv2.imwrite(f'{userimagefolder}/{name}', frame[y:y + h, x:x + w])
                    i += 1
                j += 1

        cv2.imshow('Adding new User', frame)

        if j == nimgs * 5:
            break

        if cv2.waitKey(1) == 27:  # Break on 'ESC' key
            break

    cap.release()  # Release camera resource
    cv2.destroyAllWindows()

    train_model()  # Train the model after registering a new user
    return render_template('home.html', mess=f"User {newusername} added successfully.", totalreg=totalreg())


# Convert the CSV to Excel and provide download route
@app.route('/download_excel')
def download_excel():
    csv_file = get_csv_path()
    excel_file = get_excel_path()

    # Convert CSV to Excel if CSV exists
    if os.path.exists(csv_file):
        csv_to_excel(csv_file, excel_file)

        # Check if the Excel file was successfully created
        if os.path.exists(excel_file):
            print(f"Excel file ready for download: {excel_file}")
            try:
                return send_file(excel_file, as_attachment=True)
            except Exception as e:
                return f"Error in downloading the file: {e}"
        else:
            return f"Excel file was not created!"
    else:
        return f"CSV file {csv_file} does not exist!"
# Our main function which runs the Flask App
if __name__ == '__main__':
    os.makedirs('Attendance', exist_ok=True)
    app.run(debug=True)
