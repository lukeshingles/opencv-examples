//Opencv C++ Example on Real Time Face Detection from a Video/Webcam Using Haar Cascade

/*We can similarly train our own Haar Classifier and Detect any object which we want
Only Thing is we need to load our Classifier in palce of cascade_frontalface_alt2.xml */

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

int main(void)
{
  CascadeClassifier face_cascade;
  face_cascade.load("haarcascade_frontalface_default.xml");
  if(!face_cascade.load("haarcascade_frontalface_default.xml"))
  {
    cerr<<"Error Loading XML file"<<endl;
    return 0;
  }

  VideoCapture capture(0);
  if (!capture.isOpened())
  {
    throw "Error capturing from camera.";
  }

  capture.set(CV_CAP_PROP_FRAME_WIDTH, 640);
  capture.set(CV_CAP_PROP_FRAME_HEIGHT, 480);

  namedWindow("window", 1);
  while (true)
  {
    Mat image;
    capture >> image;
    if (image.empty())
      break;

    std::vector<Rect> faces;
    face_cascade.detectMultiScale(image, faces, 1.2, 5, 0|CV_HAAR_SCALE_IMAGE, Size(10, 10));

    Mat stackedfaces;
    for(int i = 0; i < faces.size(); i++)
    {
      // Point center(faces[i].x + faces[i].width * 0.5, faces[i].y + faces[i].height * 0.5);
      // ellipse(image, center, Size( faces[i].width * 0.5, faces[i].height * 0.5), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);

      Rect facerect(faces[i].x, faces[i].y, faces[i].width, faces[i].height);

      rectangle(image, facerect, Scalar(255, 0, 0), 2);

      Mat faceresized = image(facerect);
      resize(image(facerect), faceresized, Size(256, int(256. / faces[i].width * faces[i].height)));

      if (i == 0)
        stackedfaces = faceresized;
      else
        vconcat(stackedfaces, faceresized, stackedfaces);
    }

    if (faces.size() > 0)
    {
      imshow("Faces", stackedfaces);
      moveWindow("Faces", 0, 0);
    }

    imshow("Camera", image);
    moveWindow("Camera", 256, 0);

    char c = (char) waitKey(1);
    if(c == 27) // escape key
      break;
  }

  return 0;
}
