# Adaptation of ORB_SLAM for Introspection

## YAML Parameters

### Camera Parameters

* `Camera.fx` -- `LEFT.P` (0,0) element (pixels) - focal length in the x direction. 
* `Camera.fy` -- `LEFT.P` (1,1) element (pixels) - focal length in the y direction.
    * Note that for most cameras `Camera.fx` ~= `Camera.fx` because they have square pixels.
* `Camera.cx` --  `LEFT.P` (0,2) element (pixels) - image plane origin offset in the x direction.
* `Camera.cy` -- `LEFT.P` (1,2) element (pixels) - image plane origin offset in the y direction.
    * Note that `Camera.cx` ~= `Camera.width`/2 and that `Camera.cy` ~= `Camera.height`/2.

* `Camera.k1` -- TODO - radial distortion coefficient.
* `Camera.k2` -- TODO - radial distortion coefficient.
* `Camera.p1` -- TODO - tangential distortion coefficient.
* `Camera.p2` -- TODO - tangential distortion coefficient.
    * Note that the distortion parameters are not needed here because the photos have already been undistorted in `legoloam2kitti`.

* `Camera.width` -- TODO (pixels) - camera width (i.e. the x direction).
* `Camera.height` -- TODO (pixels) - camera height (i.e. the x direction).
    * Be considerate to size the image properly with binning and NOT with cropping.
* `Camera.fps` -- TODO (frames/s) - 
* `Camera.bf` -- Stereo baseline (m) times `Camera.fx`, also the absolute value of the `RIGHT.P` (0,3) element (m*pixel) - Camera calibration calculates [R|T] between the Left and Right camera, the stereo baseline is the first element of T. 
    * Note that `Camera.bf` > 0.
* `Camera.RGB` -- Color order of the images (0: BGR, 1: RGB) - It is ignored if images are grayscale.
* `ThDepth` -- Close/far threshold (TODO) - TODO

### Calibration Parameters
TODO - do we only need these if we need to pre-rectify the images?

* `LEFT/RIGHT.height` -- TODO (pixels) - camera width (i.e. the x direction) that the camera was calibrated with.
* `LEFT/RIGHT.width` -- TODO (pixels) - camera height (i.e. the x direction) that the camera was calibrated with.
* `LEFT/RIGHT.D` -- Distortion vector (TODO) - 
    * Note this is not needed because images have already been undistorted.
* `LEFT/RIGHT.K` -- Camera matrix - "Intrinsic camera matrix for the raw (distorted) images."
* `LEFT/RIGHT.R` -- Rectifcation matrix - "A rotation matrix aligning the camera coordinate system to the ideal stereo image plane so that epipolar lines in both stereo images are parallel"
* `LEFT/RIGHT.P` -- Projection matrix - "By convention, this matrix specifies the intrinsic (camera) matrix of the processed (rectified) image."
    * Note see the [sensor_msgs::CameraInfo](https://docs.ros.org/en/api/sensor_msgs/html/msg/CameraInfo.html) message defintion for more information on each calibration parameter.

### ORB Parameters
* `ORBextractor.nFeatures` -- "Number of features per image" - 
    * Note for training data generation this value should be ~5000 and ~2000 for inference.
* `ORBextractor.scaleFactor` -- " Scale factor between levels in the scale pyramid" - 
* `ORBextractor.nLevels` -- "Number of levels in the scale pyramid  " - 
* `ORBextractor.iniThFAST` -- Intial Fast threshold - 
* `ORBextractor.minThFAST` -- Minimum Fast threshold - 
    * Note that "if no corners are detected (with the `iniThFAST` threshold) we impose a lower value `minThFAST`".
* `ORBextractor.enableIntrospection` -- -
    * Note this is a parameter added for IV-SLAM.

* `ORBmatcher.NNRatioMultiplier` -- TODO - TODO
* `ORBmatcher.SearchWindowMultiplier` -- TODO - TODO 

### Introspective Model Training Parameters
* `IVSLAM.unsupervisedLearning` -- Generate training data or not (bool) - If set to true, training data for the introspeciton model (the heatmaps) are generated in an unsupervised manner and reference camera poses are only used for evaluating the reliability of ORB-SLAM's output.

### Viewer Parameters
* `Viewer.KeyFrameSize` -- -
* `Viewer.KeyFrameLineWidth` -- -
* `Viewer.GraphLineWidth` -- -
* `Viewer.PointSize` -- -
* `Viewer.CameraSize` -- -
* `Viewer.CameraLineWidth` -- -
* `Viewer.ViewPointX` -- -
* `Viewer.ViewPointY` -- -
* `Viewer.ViewPointZ` -- -
    * Note to have a top view set to ~ -0.3
* `Viewer.ViewPointF` -- -
    * Note to have a top view set to ~ 50
* `Viewer.HeadlessMode` -- -
    * Note if set to 1, frame drawings will happen in headless mode. Useful if you want to save visualizations to file. If you want to completely turn off the viewer, you should turn it off when instantiating the SLAM object.
* `Viewer.SaveFramesToFile` -- -
* `Viewer.SaveMapDrawingsToFile` -- -

## Script Parameters

