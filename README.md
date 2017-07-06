#####  Object-Tracking-and-Detection ######

I pursued my summer intern related to image processing. The aim was to track and count objects(vehicles) in a real time scenario. I used various algorithms such as MOG Background subtraction , Haar cascading , Kalman Filtering , Blob Analysis for tracking purpose. Main objective was to keep track of information of vehicles entering and assign them proper ID's and store/write various informations such as (time stamps, area ,aspect ratio) into a file.


######## To run the code -(code.py) #######

1) Enter Path of video file at the start of program

2) Run the file and then enter the coordinates for two lines to be drawn(NOTE: Coordinates are given as a ratio b.w (0,1)).

3) User can change the parameters to filter out bad blobs by modifying the line 362 in code.py

4) At the end I have displayed the trackedlist which contains ID's of the tracked objects and according to these ID's 
   I have written the output of (time stamp, centerpositions,blobarea,blobaspect ratio) as pairs to file "info.txt" .
  


####### Opencv(3.1) Installation and its Various dependencies : ########

Follow these steps to install opencv(3.1) with python 2.7 and the required dependencies.

1) $ sudo apt-get install build-essential cmake pkg-config


2) Install Image I/O libraries -
$ sudo apt-get install libjpeg8-dev libtiff5-dev libjasper-dev libpng12-dev


3) Packages for processing video streams -
$ sudo apt-get install libavcodec-dev libavformat-dev libswscale-devlibv4l-dev
$ sudo apt-get install libxvidcore-dev libx264-dev


4) Module for handling GUI operations -
$ sudo apt-get install libgtk-3-dev


5) Library for optimising various functionalities inside opencv-
$ sudo apt-get install libatlas-base-dev gfortran


6) Install python development headers and libraries -
$ sudo apt-get install python2.7-dev python3.5-dev


7) Download the opencv source -
$ cd ~
$ wget -O opencv.zip
https://github.com/Itseez/opencv/archive/3.1.0.zip
$ unzip opencv.zip


8) Download opencv_contrib repo as well -
$ wget -O opencv_contrib.zip https://github.com/Itseez/opencv_contrib/archive/3.1.0.zip
$ unzip opencv_contrib.zip


9) Setup python environment -
$ cd ~
$ wget https://bootstrap.pypa.io/get-pip.py
$ sudo python get-pip.py


10) Install numpy ( a Python package for numerical processing)
$ pip install numpy


11) Setup and configure our build using cmake -
$ cd ~/opencv-3.1.0/
$ mkdir build
$ cd build
$ cmake -D CMAKE_BUILD_TYPE=RELEASE \
-D CMAKE_INSTALL_PREFIX=/usr/local \
-D INSTALL_PYTHON_EXAMPLES=ON \
-D INSTALL_C_EXAMPLES=OFF \
-D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib-
3.1.0/modules \
-D PYTHON_EXECUTABLE=~/.virtualenvs/cv/bin/python \
-D BUILD_EXAMPLES=ON ..

Note: If you are getting error related to stdlib.h : No such file or directory
during either cmake or make you will need to include the following option
to Cmake :
-D ENABLE_PRECOMPILED_HEADERS=OFF . In this case, I would suggest
deleting your build directory , recreating it and re-run cmake with above
options included.


12) Finally execute cmake to configure our build -
$ make clean
$ make


13) This step is to install opencv -
$ sudo make install
$ sudo ldconfig


14) After running sudo make install , your Python 2.7 bindings for OpenCV 3 should now be located in /usr/local/lib/python-2.7/site- packages/. You can verify this using the ls command -
$ ls -l /usr/local/lib/python2.7/site-packages/


15) Test your opecv installation -
$ python
$ import cv2
If this doesnt show any error then you have successfully installed opencv

$ cv2.__version__
This shows you the version of opencv installed.
