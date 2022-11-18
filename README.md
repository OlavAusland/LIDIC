# LIDIC - Live Interactive Drone Imaging Control
This repository is an implementation of machine vision and deep neural networks to
be able to predict hand gestures and control a Tello EDU drone.

The default repository recognizes five different gestures: stop, up, down, left & right.
The default model has a input shape of (42,) and a output shape of (5,) the 5 corresponding to the gestures predicted.

![](static/default_model.png)

<div style="margin: auto;width:50%">
    <img src="static/ControllerLayout.png" width="100%" alt=""/>
</div>


This repository also allow to control the drone using your keyboard, a xbox controller as well as by gestures,
which is the focus of this repository.
## Demo Video:

## Research Paper:
<iframe width="100%" height="800" src="static/LIDIC.pdf"></iframe>