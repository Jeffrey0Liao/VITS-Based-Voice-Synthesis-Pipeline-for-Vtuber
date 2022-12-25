# VITS-Based-Voice-Synthesis-Pipeline-for-Vtuber

## 1. enviroment

Our program is running on python version 3.7.

Before playing with your own vtuber you should go into env file

then

```python
pip install -r requirements.txt
```

to install the related tools and enviroments.

## 2. FaceTrack

You can just run face_track.py to see how to track your bace based on 68 keypoints. And based on the eyebrow, nose, jaw's center to calculate the rotation list for further simulating the movements of the vtuber.

## 3. vtuberDrawer

you can run vtuberDrawerLive2D.py  to read psd file and draw every layer separately.

## 4. Combination

Run virtual.py to see the combination of face track and manipulate your own vtuber