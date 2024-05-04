import pyttsx3
import speech_recognition as sr
import webbrowser
import os
import google.generativeai as genai
import winsound
import cv2
import numpy as np
import time

def beep():
    frequency = 2500
    duration = 300 
    winsound.Beep(frequency, duration)

chatStr = ""

def chat(query):
    global chatStr
    print(chatStr)

    # Configure Gemini API
    genai.configure(api_key="")

    # Initialize the GenerativeModel
    generation_config = {
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 0,
        "max_output_tokens": 8192,
    }
    safety_settings = [
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
    ]
    model = genai.GenerativeModel(
        model_name="gemini-1.5-pro-latest",
        generation_config=generation_config,
        safety_settings=safety_settings
    )

    # Start the chat with history
    convo = model.start_chat(history=[])
    convo.send_message(chatStr + "Fayez: " + query)
    response = convo.last.text

    # Update conversation history
    chatStr += f"Fayez: {query}\n Dumbo: {response}\n"

    # Print and return response
    print(response)
    return response

def ai(inputt):
    genai.configure(api_key="")

    # Set up the model
    generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 0,
    "max_output_tokens": 8192,
    }

    safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    ]

    model = genai.GenerativeModel(model_name="gemini-1.5-pro-latest",
                                generation_config=generation_config,
                                safety_settings=safety_settings)

    convo = model.start_chat(history=[])
    convo.send_message(inputt)
    print(convo.last.text)
    return convo.last.text


def say(text):
    engine = pyttsx3.init()
    speed = 150  # Speed value ranges from 0 to 200
    engine.setProperty('rate', speed)
    engine.say(text)
    engine.runAndWait()

def takeCommand():
    beep()
    print("Listening.....")
    r =sr.Recognizer()
    with sr.Microphone() as source:
        r.pause_threshold = 0.6
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)

        try:
            query = r.recognize_google(audio,language="en-PK")
            print(f"User said:{query}")
            return query
        except Exception as e:
            return "Sorry, unable to understand"

# face unlock part

def load_images_from_folder(folder):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            images.append(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))  # Convert image to grayscale
            # Extract label from filename
            label = int(filename.split('fayez')[1].split('.')[0]) 
            labels.append(label)
    return images, labels

train_images, train_labels = load_images_from_folder('images')

# Convert labels to numpy array
train_labels = np.array(train_labels)

# Initialize LBPH face recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

# Train the recognizer with the loaded images and labels
recognizer.train(train_images, train_labels)

# Load pre-trained face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# person_recognized = False


def recognize_face():
    # Initialize webcam
    cap = cv2.VideoCapture(0)

    # Start time
    start_time = time.time()
    
    person_recognized = False
    person=""
    
    while time.time() - start_time < 2:
        # Capture frame-by-frame
        ret, frame = cap.read()

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)
        
        for (x, y, w, h) in faces:
            roi_gray = gray_frame[y:y+h, x:x+w]
            label, confidence = recognizer.predict(roi_gray)
            if confidence > 70: 
                if not person_recognized:
                    person_recognized = False
            else:
                # If face is not recognized
                if not person_recognized:
                    person_recognized = True 
                
            # Draw rectangle around the face
            color = (255, 0, 0)
            thickness = 2
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness)

        # Display the resulting frame
        cv2.imshow('frame', frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
    
    return person_recognized





if __name__ == '__main__':

    flg=False
    flg = recognize_face()

    if not flg:
        say("Access granted")
        print("Security unlocked")

        say("Hello and welcome fayez, i am your AI assistant Dumboo")

        while True:

            query = takeCommand()
            sites =[["youtube","https://youtube.com"],["Google","https://Google.com"]]

            for site in sites:

                if f"open {site[0]}".lower() in query.lower():
                    say(f"Opening {site[0]}...")
                    webbrowser.open(site[1])
            c=0
            ai_response=""
            if "turn on ai mode".lower() in query.lower():
                say("AI mode turning on")
                while("turn off ai mode".lower() not in query.lower()):
                    if(c==0):
                        query = takeCommand()
                    c +=1
                    ai_response = ai(query)
                    # say(ai_response)
                    query = takeCommand()
                say("AI mode truning off")
            if "Shutdown the system".lower() in query.lower():
                say("system is powering off, bye bye")
                exit()

            else:
                print("Chatting  :)")
                chat(query)
    else:
        say("Access denied, you are unauthorized person")
        print("Unknown person")

