/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/


#ifndef TRACKING_H
#define TRACKING_H

#include <glog/logging.h>
#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#if (CV_VERSION_MAJOR >= 4)
  #include<opencv2/imgproc/imgproc_c.h>
#endif

#include"Viewer.h"
#include"FrameDrawer.h"
#include"Map.h"
#include"LocalMapping.h"
#include"LoopClosing.h"
#include"Frame.h"
#include "ORBVocabulary.h"
#include"KeyFrameDatabase.h"
#include"ORBextractor.h"
#include "Initializer.h"
#include "MapDrawer.h"
#include "System.h"
#include "feature_evaluator.h"
#include "dataset_creator.h"
#include "io_access.h"

#include <mutex>

namespace ORB_SLAM2
{

class Viewer;
class FrameDrawer;
class Map;
class LocalMapping;
class LoopClosing;
class System;

class Tracking
{  

public:
    Tracking(System* pSys, 
             ORBVocabulary* pVoc, 
             FrameDrawer* pFrameDrawer, 
             MapDrawer* pMapDrawer, 
             Map* pMap,
             KeyFrameDatabase* pKFDB, 
             const string &strSettingPath, 
             const int sensor,
             const bool bSingleThreaded=false,
             const bool bSilent=false,
             const bool bGuidedBA=false);

    // Preprocess the input and call Track(). Extract features and performs 
    // stereo matching.
    cv::Mat GrabImageStereo(const cv::Mat &imRectLeft,
                            const cv::Mat &imRectRight, 
                            const double &timestamp);
    
    cv::Mat GrabImageStereo(const cv::Mat &imRectLeft,
                        const cv::Mat &imRectRight, 
                        const double &timestamp,
                        const cv::Mat &cam_pose_gt,
                        const Eigen::Matrix<double, 6, 6>& cam_pose_gt_cov,
                        const bool &pose_cov_available,
                        const std::string& img_name,
                        const bool &gtDepthAvailable = false,
                        const cv::Mat &depthmap = cv::Mat(0,0,CV_32F));
    
    cv::Mat GrabImageRGBD(const cv::Mat &imRGB,const cv::Mat &imD, const double &timestamp);
    cv::Mat GrabImageMonocular(const cv::Mat &im, const double &timestamp);
    
    // The ground truth depth map is also an optional argument that could be
    // passed to this function. In order for imDepth to actually be used as 
    // ground truth depth gtDepthAvailable should be set to true. If 
    // gtDepthAvailable is "false" and imDepth is provided, it is assumed
    // that it is a predicted image quality heatmap to be used for scoring
    // the extracted keypoints.
    cv::Mat GrabImageMonocular(const cv::Mat &im,
                               const double &timestamp,
                               const cv::Mat &cam_pose_gt,
                               const std::string& img_name,
                               const bool &gtDepthAvailable = false,
                               const cv::Mat &depthmap = cv::Mat(0,0,CV_32F));

    void SetLocalMapper(LocalMapping* pLocalMapper);
    void SetLoopClosing(LoopClosing* pLoopClosing);
    void SetViewer(Viewer* pViewer);

    // Load new settings
    // The focal lenght should be similar or scale prediction will fail when projecting points
    // TODO: Modify MapPoint::PredictScale to take into account focal lenght
    void ChangeCalibration(const string &strSettingPath);

    // Use this function if you have deactivated local mapping and you only want to localize the camera.
    void InformOnlyTracking(const bool &flag);
    
    // Set the list of relative camera pose uncertainty for all frames. This
    // should only be called in training mode and if such information is 
    // available
    void SetRelativeCamPoseUncertainty(
        const  std::unordered_map<std::string, int>* pose_unc_map,
        const vector<Eigen::Vector2f>* rel_cam_poses_uncertainty);
    
    cv::Mat CalculateInverseTransform(const cv::Mat& transform);
    
    // Release all allocated memory
    void Release();
  

public:

    // Tracking states
    enum eTrackingState{
        SYSTEM_NOT_READY=-1,
        NO_IMAGES_YET=0,
        NOT_INITIALIZED=1,
        OK=2,
        LOST=3
    };

    eTrackingState mState;
    eTrackingState mLastProcessedState;

    // Input sensor
    int mSensor;
    
    // Are tracking and local mapping running a single thread
    bool mbSingleThreaded;
    
    // In single threaded mode, number of times local bundle adjustment is 
    // called is limited by this threshold. Every mRateBAInSingleThreadedMode
    // frames local BA is called once.
    int mFramesReceivedSinceLastLocalBA = 0;

    // Current Frame
    Frame mCurrentFrame;
    cv::Mat mImGray;
    
    // Previous captured image of left camera: only used for visualizations
    // and debugging in training mode
    cv::Mat mImGrayPrev;
    
    // Current frame's ground truth depth map (obtained in simulation). Only 
    // used for debugging and verification
    cv::Mat mImDepth;

    // Initialization Variables (Monocular)
    std::vector<int> mvIniLastMatches;
    std::vector<int> mvIniMatches;
    std::vector<cv::Point2f> mvbPrevMatched;
    std::vector<cv::Point3f> mvIniP3D;
    Frame mInitialFrame;

    // Lists used to recover the full camera trajectory at the end of the execution.
    // Basically we store the reference keyframe for each frame and its relative transformation
    list<cv::Mat> mlRelativeFramePoses;
    list<KeyFrame*> mlpReferences;
    list<double> mlFrameTimes;
    list<bool> mlbLost;
    
    // List of estimated camera poses. It is different from the above relative
    // poses, in that these stored values are the estimated pose for each 
    // frame right after it was processed (and BA was run if you are running 
    // ORB-SLAM tracking in single threaded mode). Hence it does not take 
    // into account adjustments in the estimated pose of a frame once BA is 
    // run on the proceeding frames. Poses are stored as pairs of frame ID and
    // camera pose.
//     std::unordered_map<int, cv::Mat> mvFramePoses;
    
    // Estimated camera pose for the latest received frame
    cv::Mat mCurrentFramePose;
    
    // Keeps count of number of times tracking has got lost
    int mFailureCount = 0;

    // Attributes regarding the introspection model
    // True if data is being collected for training the introspective model
    bool mbTrainingMode = false;
    
    // True if unsupervised training data is being generated for image feature
    // scores. In this mode, instead of relying on the epipolar or reprojection
    // errors estimated based on the "ground truth" pose of the camera, reproj 
    // error calculated based on the "estimated" pose of the camera is used for 
    // evaluating image features.
    bool mbUnsupervisedLearning = false;
    
    // If set to true, supervised feature evaluation will happen even alongside
    // unsupervised learning. This is mainly for logging extracted features' 
    // ground truth information for evaluation purposes. Keep this off in
    // normal operation and if generating a training dataset.
    const bool mbEnforceSupervisedFeatureEval = false;
    
    // True if introspection model is engaged
    bool mbIntrospectionOn = false;
    
    bool mbCreateIntrospectionDataset = false;
    // The path to the generated dataset for training the introspection model
    // It only includes images that are selected to be balanced for training
    std::string mvOutputIntrospectionDatasetPath;
    // The path to the datast that includes all images and not only those
    // that are selected to be used for training
    std::string mvOutputIntrospectionDatasetFullPath;
    
    int iLoggingLevel = 0;
    std::string mvSaveVisualizationPath;
    
    
    
    // True if local mapping is deactivated and we are performing only localization
    bool mbOnlyTracking;
   
    // Flags tracker to save the stored data to file. Should be called before
    // shutdown
    void SaveIntrospectionDataset();
    
    // Save the tracked keyframe poses to file. It also saves the time stamps
    // of tracking failures (This is especially used when in experimental
    // mode where we reset tracking each time it is lost.)
    // The argument is set to true if results are being saved right after 
    // a failure happening to log accordingly
    void SaveTrackingResults(bool saving_on_failure);

    
    void Reset();

private:
    float mMatcherNNRatioMultiplier = 1.0;
    float mMatcherSearchWindowMultiplier = 1.0;

protected:

    // Main tracking function. It is independent of the input sensor.
    void Track();

    // Map initialization for stereo and RGB-D
    void StereoInitialization();

    // Map initialization for monocular
    void MonocularInitialization();
    void CreateInitialMapMonocular();

    void CheckReplacedInLastFrame();
    
    // If use_BoW is set to false, it performs search by projection with 
    // relaxed constraints instead
    bool TrackReferenceKeyFrame(float nn_ratio_mult = 1.0, 
                                float search_wind_mult = 1.0,
                                bool use_BoW = true);
    void UpdateLastFrame();
    bool TrackWithMotionModel(float nn_ratio_mult = 1.0, 
                              float search_wind_mult = 1.0);

    bool Relocalization();

    void UpdateLocalMap();
    void UpdateLocalPoints();
    void UpdateLocalKeyFrames();

    bool TrackLocalMap();
    void SearchLocalPoints();

    bool NeedNewKeyFrame();
    void CreateNewKeyFrame();
    
    // Visualize estimated quality scores of they keypoints of the latest
    // keyframe
    void VisualizeKeyPtsQuality();
    
    // Evaluates the current tracking accuracy by comparing against refererence
    // camera poses. Returns true if tracking is reliable and false otherwise.
    bool EvaluateTrackingAccuracy();
    

    // In case of performing only localization, this flag is true when there are no matches to
    // points in the map. Still tracking will continue if there are enough matches with temporal points.
    // In that case we are doing visual odometry. The system will try to do relocalization to recover
    // "zero-drift" localization to the map.
    bool mbVO;

    //Other Thread Pointers
    LocalMapping* mpLocalMapper;
    LoopClosing* mpLoopClosing = NULL;

    //ORB
    ORBextractor* mpORBextractorLeft, *mpORBextractorRight;
    ORBextractor* mpIniORBextractor;

    //BoW
    ORBVocabulary* mpORBVocabulary;
    KeyFrameDatabase* mpKeyFrameDB;

    // Initalization (only for monocular)
    Initializer* mpInitializer;

    //Local Map
    KeyFrame* mpReferenceKF;
    std::vector<KeyFrame*> mvpLocalKeyFrames;
    std::vector<MapPoint*> mvpLocalMapPoints;
    
    // System
    System* mpSystem;
    
    //Drawers
    Viewer* mpViewer;
    FrameDrawer* mpFrameDrawer=NULL;
    MapDrawer* mpMapDrawer;

    //Map
    Map* mpMap;

    //Calibration matrix
    cv::Mat mK;
    cv::Mat mDistCoef;
    float mbf;

    //New KeyFrame rules (according to fps)
    int mMinFrames;
    int mMaxFrames;

    // Threshold close/far points
    // Points seen as close by the stereo/RGBD sensor are considered reliable
    // and inserted from just one frame. Far points requiere a match in two keyframes.
    float mThDepth;

    // For RGB-D inputs only. For some datasets (e.g. TUM) the depthmap values are scaled.
    float mDepthMapFactor;

    //Current matches in frame
    int mnMatchesInliers;

    //Last Frame, KeyFrame and Relocalisation Info
    KeyFrame* mpLastKeyFrame;
    Frame mLastFrame;
    unsigned int mnLastKeyFrameId;
    unsigned int mnLastRelocFrameId;

    //Motion Model
    cv::Mat mVelocity;

    //Color order (true RGB, false BGR, ignored if grayscale)
    bool mbRGB;

    list<MapPoint*> mlpTemporalPoints;
    
    //Feature evaluator for preparing training data for the introspection model
    feature_evaluation::FeatureEvaluator* mFeatureEvaluator;
    
    //Dataset creator saves information extracted by feature evaluator to file
    // in the format expected by the introspection network trainer.
    feature_evaluation::DatasetCreator* mDatasetCreator = NULL;
    
    // This one creates a dataset of all images and not only those
    // that are selected to be used for training
    feature_evaluation::DatasetCreator* mDatasetCreatorFull = NULL;
    
    // Set the list of relative camera pose uncertainty for all frames. 
    bool mbRelCamPoseUncertaintyAvailable = false;
    const std::unordered_map<std::string, int>* mpPoseUncMap;
    const std::vector<Eigen::Vector2f>* mvpRelPoseUnc;
    
    // Almost no print outs in silent mode. Also the tracked trajectory is not
    // saved to file upon shutdown
    bool mbSilent;
    
    // If set to true and in training mode, camera posese are fixed to 
    // ground truth values. This is to evaluate map points given their 
    // reprojection error
    bool mbGuidedBA;
};

} //namespace ORB_SLAM

#endif // TRACKING_H
