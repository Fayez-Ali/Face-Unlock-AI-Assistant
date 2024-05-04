# Face-Unlock-AI-Assistant


Sure, here's a breakdown of the project along with a detailed description of each module:

1. **Face Unlock Module**:
   - This module is responsible for recognizing the authorized user's face using the LBPH (Local Binary Patterns Histograms) Face Recognizer.
   - It utilizes the OpenCV library to capture frames from the webcam and detect faces using the pre-trained Haar Cascade Classifier.
   - The LBPH Face Recognizer is trained on a dataset of authorized user images to recognize the user's face.
   - If the confidence level of recognition is above a certain threshold, access is granted; otherwise, access is denied.

2. **Voice Assistant Module**:
   - This module consists of functions for speech recognition, text-to-speech conversion, and natural language processing.
   - The `takeCommand()` function listens to user commands using the microphone and converts speech to text.
   - The `say()` function converts text to speech and responds to the user's queries or commands.
   - It uses the SpeechRecognition and pyttsx3 libraries for speech recognition and text-to-speech conversion, respectively.
   - The `chat()` function interacts with the user by responding to queries using the Generative AI model (Gemini) from Google GenAI.

3. **AI Mode**:
   - This mode enables the AI assistant to engage in conversational interactions with the user.
   - When activated, the assistant responds to user queries and engages in a chat using the Gemini Generative AI model.
   - The AI mode can be toggled on and off by the user's voice command.
   - The Gemini model generates responses based on the user's input and previous conversation history.

4. **Web Browser Control**:
   - The assistant can open web browser tabs based on user commands.
   - It uses the webbrowser library to open URLs in the default web browser.

5. **Open Apps**:
   - The assistant can open apps based on user commands.
   - It uses the os library to open apps whose path is saved in the code.

6. **System Shutdown**:
   - The assistant can shut down the system based on user command.
   - It listens for the voice command "Shutdown the system" and initiates the system shutdown process.

7. **Main Functionality**:
   - The `main()` function orchestrates the entire workflow of the AI assistant.
   - It first checks if the face unlock module grants access to the authorized user.
   - If access is granted, the assistant welcomes the user and enters into a conversational mode.
   - In conversational mode, the assistant listens to user queries, performs actions (such as opening websites), engages in chat interactions, and handles system shutdown commands.

8. **Project Workflow**:
   - The project workflow starts with the user's face being recognized by the face unlock module.
   - Upon successful recognition, the AI assistant initializes and welcomes the user.
   - The assistant listens to user commands and queries through speech recognition.
   - It responds to user queries using text-to-speech conversion and interacts with the user in conversational mode.
   - The assistant can perform actions such as opening web browser tabs, engaging in chat conversations using the Gemini model, and shutting down the system.
   - The workflow continues until the user exits the application or initiates system shutdown.

Overall, the project integrates face recognition technology with voice-based AI assistance to create a secure and interactive user experience. It demonstrates the capabilities of AI in providing personalized assistance and engaging in natural language conversations with users.
