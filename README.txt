INSTRUCTIONS:
[PLEASE ONLY ONE FACE AT A TIME]
Step 1: The boost library was too big to include so you will need to redo it. The instructions for this can be found near the top of this article 'http://www.paulvangent.com/2016/08/05/emotion-recognition-using-facial-landmarks/'
(optional) If you added new pictures and wish to retrain just delete the 'trained_svm_model' file from the source folder and run 'TrainModel.py'.
(optional) If you wish to create new training data run 'SaveYourFaces'. A window will pop up and when a green box appears around your face it will save a cropped ready to use image to the custom folder. Just drag it to the correct emotion sub-folder in the dataset folder.
(From Image) Open 'PredictFromImage.py' and change the directory to an image of your choosing and run it. All the results will appear in the console.
(Live) Just run 'PredictFromWebcam.py'.

The below articles helped create the foundation for this project.

CITATIONS:
van Gent, P. (2016). Emotion Recognition With Python, OpenCV and a Face Dataset. A tech blog about fun things with Python and embedded electronics. Retrieved from: http://www.paulvangent.com/2016/04/01/emotion-recognition-with-python-opencv-and-a-face-dataset/
van Gent, P. (2016). Emotion Recognition Using Facial Landmarks, Python, DLib and OpenCV. A tech blog about fun things with Python and embedded electronics. Retrieved from: http://www.paulvangent.com/2016/08/05/emotion-recognition-using-facial-landmarks/