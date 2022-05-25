from tkinter import *
from PIL import Image,ImageTk
import speech_recognition as sr
import easyimap as e
import pyttsx3
import smtplib
from email.message import EmailMessage
import face_recognition
import cv2
import pickle
import os
import numpy as np
from pathlib import Path
r = sr.Recognizer()
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)
engine.setProperty('rate', 150)

def speak(s):
    print(s)
    engine.say(s)
    engine.runAndWait()

def listen():
    with sr.Microphone() as source:
        r.adjust_for_ambient_noise(source)
        s = "Speak Now:"
        speak(s)
        audio = r.listen(source)
        try:
            text = r.recognize_google(audio)

            return text
        except:
            s = "Sorry could not recognize what you said";
            speak(s)
            #speak("try again")
            #listen()

# class: Dlib Face Unlock
# Purpose: This class will update the encoded known face if the directory has changed
# as well as encoding a face from a live feed to compare the face to allow the facial recognition
# to be integrated into the system
# Methods: ID
class Dlib_Face_Unlock:
    # When the Dlib Face Unlock Class is first initialised it will check if the employee photos directory has been
    # updated if an update has occurred either someone deleting their face from the system or someone adding their
    # face to the system the face will then be encoded and saved to the encoded pickle file
    def __init__(self):
        # this is to detect if the directory is found or not
        try:
            # this will open the existing pickle file to load in the encoded faces of the users who has sign up for
            # the service
            with open(r'C:\Users\Divya Mathew\PycharmProjects\CommunityProject\labels.pickle', 'rb') as self.f:
                self.og_labels = pickle.load(self.f)
            print(self.og_labels)
        # error checking
        except FileNotFoundError:
            # allowing me to known that their was no file found
            print("No label.pickle file detected, will create required pickle files")

        # this will be used to for selecting the photos
        self.current_id = 0
        # creating a blank ids dictionary
        self.labels_ids = {}
        # this is the directory where all the users are stored
        self.BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.image_dir = os.path.join(self.BASE_DIR, 'images')
        for self.root, self.dirs, self.files in os.walk(self.image_dir):
            # checking each folder in the images directory
            for self.file in self.files:
                # looking for any png or jpg files of the users
                if self.file.endswith('png') or self.file.endswith('jpg'):
                    # getting the folder name, as the name of the folder will be the user
                    self.path = os.path.join(self.root, self.file)
                    self.label = os.path.basename(os.path.dirname(self.path)).replace(' ', '-').lower()
                    if not self.label in self.labels_ids:
                        # adding the user into the labels_id dictionary
                        self.labels_ids[self.label] = self.current_id
                        self.current_id += 1
                        self.id = self.labels_ids[self.label]

        print(self.labels_ids)
        # this is compare the new label ids to the old label ids dictionary seeing if their has been any new users or old users
        # being added to the system, if there is no change then nothing will happen
        self.og_labels = 0
        if self.labels_ids != self.og_labels:
            # if the dictionary change then the new dictionary will be dump into the pickle file
            with open('labels.pickle', 'wb') as self.file:
                pickle.dump(self.labels_ids, self.file)

            self.known_faces = []
            for self.i in self.labels_ids:
                # Get number of images of a person
                noOfImgs = len([filename for filename in os.listdir('images/' + self.i)
                                if os.path.isfile(os.path.join('images/' + self.i, filename))])
                print(noOfImgs)
                for imgNo in range(1, (noOfImgs + 1)):
                    self.directory = os.path.join(self.image_dir, self.i, str(imgNo) + '.png')
                    self.img = face_recognition.load_image_file(self.directory)
                    self.img_encoding = face_recognition.face_encodings(self.img)[0]
                    self.known_faces.append([self.i, self.img_encoding])
            print(self.known_faces)
            print("No Of Imgs" + str(len(self.known_faces)))
            with open('KnownFace.pickle', 'wb') as self.known_faces_file:
                pickle.dump(self.known_faces, self.known_faces_file)
        else:
            with open(r'C:\Users\Divya Mathew\PycharmProjects\CommunityProject\KnownFace.pickle', 'rb') as self.faces_file:
                self.known_faces = pickle.load(self.faces_file)
            print(self.known_faces)

    # Method: ID
    # Purpose:This is method will be used to create a live feed .i.e turning on the device's camera
    # then the live feed will be used to get an image of the user and then encode the users face
    # once the users face has been encoded then it will be compared to in the known faces
    # therefore identifying the user
    def ID(self):
        # turning on the camera to get a photo of the user frame by frame
        self.cap = cv2.VideoCapture(0)
        # seting the running variable to be true to allow me to known if the face recog is running
        self.running = True
        self.face_names = []
        while self.running == True:
            # taking a photo of the frame from the camera
            self.ret, self.frame = self.cap.read()
            # resizing the frame so that the face recog module can read it
            self.small_frame = cv2.resize(self.frame, (0, 0), fx=0.5, fy=0.5)
            # converting the image into black and white
            self.rgb_small_frame = self.small_frame[:, :, ::-1]
            if self.running:
                # searching the black and white image for a face
                self.face_locations = face_recognition.face_locations(self.frame)

                # if self.face_locations == []:
                #     Dlib_Face_Unlock.ID(self)
                # it will then encode the face into a matrix
                self.face_encodings = face_recognition.face_encodings(self.frame, self.face_locations)
                # creating a names list to append the users identify into
                self.face_names = []
                # looping through the face_encoding that the system made
                for self.face_encoding in self.face_encodings:
                    # looping though the known_faces dictionary
                    for self.face in self.known_faces:
                        # using the compare face method in the face recognition module
                        self.matches = face_recognition.compare_faces([self.face[1]], self.face_encoding)
                        print(self.matches)
                        self.name = 'Unknown'
                        # compare the distances of the encoded faces
                        self.face_distances = face_recognition.face_distance([self.face[1]], self.face_encoding)
                        # uses the numpy module to comare the distance to get the best match
                        self.best_match = np.argmin(self.face_distances)
                        print(self.best_match)
                        print('This is the match in best match', self.matches[self.best_match])
                        if self.matches[self.best_match] == True:
                            self.running = False
                            self.face_names.append(self.face[0])
                            break
                        next
            print("The best match(es) is" + str(self.face_names))
            self.cap.release()
            cv2.destroyAllWindows()
            break
        return self.face_names

def register():
    # Create images folder
    s="What is your name?"
    speak(s)
    name=listen()
    gui(name)
    if not os.path.exists("images"):
        os.makedirs("images")
    # Create folder of person (IF NOT EXISTS) in the images folder
    Path("images/" + name).mkdir(parents=True, exist_ok=True)
    # Obtain the number of photos already in the folder
    numberOfFile = len([filename for filename in os.listdir('images/' + name)
                        if os.path.isfile(os.path.join('images/' + name, filename))])
    # Add 1 because we start at 1
    numberOfFile += 1
    # Take a photo code
    cam = cv2.VideoCapture(0)

    cv2.namedWindow("CAPTURE")

    while True:
        ret, frame = cam.read()
        cv2.imshow("test", frame)
        if not ret:
            break
        k = cv2.waitKey(1)

        if k % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            cam.release()
            cv2.destroyAllWindows()
            break
        elif k % 256 == 32:
            # SPACE pressed
            img_name = str(numberOfFile) + ".png"
            cv2.imwrite(img_name, frame)
            print("{} written!".format(img_name))
            os.replace(str(numberOfFile) + ".png", "images/" + name.lower() + "/" + str(numberOfFile) + ".png")
            cam.release()
            cv2.destroyAllWindows()
            break

# Passing in the model
def login():
    # After someone has registered, the face scanner needs to load again with the new face
    dfu = Dlib_Face_Unlock()
    # Will return the user's name as a list, will return an empty list if no matches
    user = dfu.ID()
    s="You are successfully logged in."
    speak(s)

    if user==[]:
        s= "Face Not Recognised"
        return

email_list = {
    'Jigyasa': 'jigyasabisht0789@gmail.com',
    'Divya': 'divyamathew246@gmail.com',
}


def sendmail():
    s='To Whom you want to send email'
    speak(s)
    name = listen()
    receiver = email_list[name]
    print(receiver)
    s='What is the subject of your email?'
    speak(s)
    subject = listen()
    print(subject)
    s='Tell me the text in your email'
    speak(s)
    message = listen()
    print(message)
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    # Make sure to give app access in your Google account
    server.login('emailassistant8@gmail.com', 'email@123')
    email = EmailMessage()
    email['From'] = 'emailassistant8@gmail.com';
    email['To'] = receiver
    email['Subject'] = subject
    email.set_content(message)
    server.send_message(email)
    mailgui("emailassistant8@gmail.com",receiver,subject,message)
    s = 'Your email is sent'
    speak(s)

def readmail():
    p = "email@123"
    u = "emailassistant8@gmail.com"
    server = e.connect("imap.gmail.com", u, p)
    server.listids()
    s= "Please say the serial number of the email you wanna read starting from the latest"
    speak(s)
    a = listen()
    gui(a)
    if (a == "Tu"):
        a = "2"
    b = int(a) - 1
    b = int(b)
    email = server.mail(server.listids()[b])
    s = "The email is from: "
    speak(s)
    speak(email.from_addr)
    s = "The title of mail is: "
    speak(s)
    speak(email.title)
    s = "The body of the mail is"
    speak(s)
    speak(email.body)
    mailgui(email.from_addr,"emailassistant8@gmail.com",email.title,email.body)

def mailFunctionality():
    while (1):
        s = "what do you want to do?"
        speak(s)
        s = "Speak SEND to Send email Speak READ to Read Inbox  Speak EXIT To Exit"
        speak(s)
        jigyasa()
        ch = listen()
        gui(ch)
        if (ch == "send"):
            s = "You have chosen to send an email"
            speak(s)
            sendmail()

        elif (ch == 'read'):
            s = "You have chosen to read email"
            speak(s)
            readmail()

        elif (ch == 'exit'):
            s = "You have chosen to exit. Bye bye"
            speak(s)
            break
        else:
            s = "Invalid choice, you said"
            speak(s)

def gui(s):
    from tkinter import messagebox
    root = Tk()
    root.title("login")
    root.geometry('925x500+100+100')
    root.configure(bg="#fff")
    root.resizable(False, False)
    bg = ImageTk.PhotoImage(file="speechnew.png");
    bg_Image = Label(root, image=bg).place(x=40, y=120);
    frame1 = Frame(root, width=800, height=80, bg="white")
    frame1.place(x=1, y=1)
    heading = Label(frame1, text="Welcome to voice based email service", fg='#57a1f8', bg='white',font=('Microsoft YaHei UI Light', 23, 'bold'))
    heading.place(x=80, y=1)
    frame2 = Frame(root, width=350, height=350, bg="white")
    frame2.place(x=550, y=100)
    Message(frame2, width=300, pady=7, text=s, fg='black', bg='#57a1f8',font=('Goudy old style', 30, 'bold')).place(x=100, y=15)
    root.after(2000, lambda: root.destroy())
    root.mainloop()

def mailgui(d,a,b,c):
    from tkinter import messagebox
    from PIL import ImageTk;
    root = Tk()
    root.title("mail")
    root.geometry('925x500+100+100')
    root.configure(bg="#fff")
    root.resizable(False, False)
    bg = ImageTk.PhotoImage(file="login.png");
    bg_Image = Label(root, image=bg).place(x=480, y=100);
    frame1 = Frame(root, width=800, height=80, bg="white")
    frame1.place(x=1, y=1)
    heading = Label(frame1, text="Welcome to voice based email service", fg='#57a1f8', bg='white',
                    font=('Microsoft YaHei UI Light', 23, 'bold'))
    heading.place(x=80, y=1)

    frame2 = Frame(root, width=450, height=500, bg="white")
    frame2.place(x=10, y=100)

    Button(frame2, width=7, pady=7, text="FROM", bg='black', fg='white',
           font=('Microsoft YaHei UI Light', 10, 'bold')).place(x=20, y=5)
    Button(frame2, width=7, pady=7, text="TO", bg='black', fg='white',
           font=('Microsoft YaHei UI Light', 10, 'bold')).place(x=20, y=55)
    Button(frame2, width=7, pady=7, text="SUBJECT", bg='black', fg='white',
           font=('Microsoft YaHei UI Light', 10, 'bold')).place(x=20, y=105)
    Button(frame2, width=7, pady=7, text="BODY", bg='black', fg='white',
           font=('Microsoft YaHei UI Light', 10, 'bold')).place(x=20, y=155)

    Message(frame2, width=300, pady=7, text=d, bg='#57a1f8', fg='white',
            font=('Microsoft YaHei UI Light', 10, 'bold')).place(x=100, y=5)
    Message(frame2, width=300, pady=7, text=a, bg='#57a1f8', fg='white',
            font=('Microsoft YaHei UI Light', 10, 'bold')).place(x=100, y=55)
    Message(frame2, width=300, pady=7, text=b, bg='#57a1f8', fg='white',
            font=('Microsoft YaHei UI Light', 10, 'bold')).place(x=100, y=105)
    Message(frame2, width=300, pady=7,
            text=c,
            bg='#57a1f8', fg='white',
            font=('Microsoft YaHei UI Light', 10, 'bold')).place(x=100, y=155)
    root.after(4000, lambda: root.destroy())
    root.mainloop()

def rl():
    from tkinter import messagebox
    root = Tk()
    root.title("home")
    root.geometry('925x600+100+100')
    root.configure(bg="#fff")
    root.resizable(False, False)
    bg = ImageTk.PhotoImage(file="image.png");
    bg_Image = Label(root, image=bg).place(x=70, y=120);
    frame1 = Frame(root, width=800, height=50, bg="white")
    frame1.place(x=90, y=1)
    heading = Label(frame1, text="Welcome to voice based email service", fg='#57a1f8', bg='white',
                    font=('Microsoft YaHei UI Light', 30, 'bold'))
    heading.place(x=20, y=1)
    frame2 = Frame(root, width=350, height=350, bg="white")
    frame2.place(x=480, y=100)
    Button(frame2, width=38, pady=7, text="REGISTER", bg='#57a1f8', fg='white',
           font=('Microsoft YaHei UI Light', 10, 'bold')).place(x=35, y=200)
    Button(frame2, width=38, pady=7, text="LOG IN", bg='#57a1f8', fg='white',
           font=('Microsoft YaHei UI Light', 10, 'bold')).place(x=35, y=140)
    root.after(2000, lambda: root.destroy())
    root.mainloop()

def jigyasa():
    from PIL import ImageTk;
    root = Tk()
    root.title("home")
    root.geometry('925x600+100+100')
    root.configure(bg="#fff")
    root.resizable(False, False)

    bg = ImageTk.PhotoImage(file="login.png");
    bg_Image = Label(root, image=bg).place(x=480, y=100);
    frame1 = Frame(root, width=800, height=80, bg="white")
    frame1.place(x=1, y=1)
    heading = Label(frame1, text="Welcome to voice based email service", fg='#57a1f8', bg='white',
                    font=('Microsoft YaHei UI Light', 23, 'bold'))
    heading.place(x=80, y=1)
    frame2 = Frame(root, width=470, height=500, bg="white")
    frame2.place(x=10, y=100)
    heading2 = Label(frame2, text="Speak SEND to Send email \n Speak READ to Read Inbox  \n Speak EXIT To Exit",
                     fg='black', bg='#57a1f8',
                     font=('Goudy old style', 23, 'bold'))
    heading2.place(x=20, y=10)
    root.after(3000, lambda: root.destroy())
    root.mainloop()


#start

s="Welcome to voice controlled email service"
speak(s)
s="Choose between register or login"
speak(s)
rl()
ch = listen()
gui(ch)


if(ch=="register"):
    register()
    s="You have successfully registered."
    speak(s)
    login()
    mailFunctionality()

elif(ch=="login"):
    login()
    mailFunctionality()
else:
    s="Invalid choice"
    speak(s)


