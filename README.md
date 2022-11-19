<img style='margin:0;' style="width: 100%;height:auto;" src="static/UIA_Header_English.png" alt=""/>

# LIDIC - Live Interactive Drone Imaging Control
This repository is an implementation of machine vision and deep neural networks to
be able to predict hand gestures and control a Tello EDU drone.

The default repository recognizes five different gestures: stop, up, down, left & right.
The default model has an input shape of (42,) and a output shape of (5,) the 5 corresponding to the gestures predicted.

![](static/default_model.png)

<div style="margin: auto;width:50%;">
    <img style="width:auto;" src="static/ControllerLayout.png" width="50%" alt=""/>
</div>

```bash
python tello.py -c xbox_controller
```

This repository also allow to control the drone using your keyboard, an xbox controller as well as by gestures,
which is the focus of this repository.




Gestures our model is trained on:
<div style="display:flex;flex-wrap: wrap; justify-content: center; justify-self: center">
    <img style='width:30%;' src="static/directions/UpGesture.png" alt="up gesture"/>
    <img style='width:30%;' src="static/directions/LeftGesture.png" alt="left gesture"/>
    <img style='width:30%;' src="static/directions/RightGesture.png" alt="right gesture"/>
    <img style='width:45%;' src="static/directions/DownGesture.png" alt="down gesture"/>
    <img style='width:45%;' src="static/directions/StopGesture.png" alt="stop gesture"/>
</div>

```bash
python tello.py -c gesture
```

## Installation & Use:
```bash
pip install -r requirements.txt
```

```bash
pytthon tello.py -c [keyboard, gesture, xbox_controller]
```

## Demo Video:

## [Research Paper](./static/LIDIC.pdf)