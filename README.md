# VideoClassification

I have cloned a git repo: https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/label_image 

This link contains a piece of code that reads an image and model file and does a simple classification task. 
I had modified this code from image classification to video classification. The input was a video. When the video is played, it gives a label text on display in real-time.

The key factors of my project:
When the confidence level is above 80%, the text result is shown in green.
Otherwise, when the confidence level is below 80%, the text result is shown in red.
