This project involved learning about the basics of deep learning through various exercises. The culmination of this project was a Deep Neural Network that provided steering angles and throttle values to a virtual car in a Unity simulation. The aforementioned model was trained on images collected whilst operating the car in the simulation manually. As such, it was a successful implementation of Behavioural Cloning. Furthermore, all exercises employed Tensorflow, and more specifically the Keras API.
The steps taken to achieve this project included fundamental study of Deep Learning basics, starting from simple linear classification and regression, and building up to Convolutional Neural Networks.
This repo includes:
    • Lane finding algorithm using edge detection
    • CNN that classifies MNIST handwritten digits
    • CNN that classifies German traffic signs (approx 40 signs w/ an accuracy of ~.96)
    • Final Project: Behavioural Cloning model that drives a car in a Unity sim. Accomplished using Flask and SocketIO to establish bidirectional communication between the simulation and the “Drive.py” script, which sends “steering angles” and “throttle values” to the simulation based on receiving input from cameras on the car. Car was trained and validated on the first track, and was tested on a second, unseen track. Performance was good, with no crashes and the car successfully completing the course. 

