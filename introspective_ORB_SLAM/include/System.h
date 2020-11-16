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


#ifndef SYSTEM_H
#define SYSTEM_H

#include<string>
#include<thread>
#include<opencv2/core/core.hpp>
#include <unistd.h>


#include "Tracking.h"
#include "FrameDrawer.h"
#include "MapDrawer.h"
#include "Map.h"
#include "LocalMapping.h"
#include "LoopClosing.h"
#include "KeyFrameDatabase.h"
#include "ORBVocabulary.h"
#include "Viewer.h"

namespace ORB_SLAM2
{

class Viewer;
class FrameDrawer;
class Map;
class Tracking;
class LocalMapping;
class LoopClosing;

class System
{
public:
    // Input sensor
    enum eSensor{
        MONOCULAR=0,
        STEREO=1,
        RGBD=2
    };

public:
    // Initialize the SLAM system. It launches the Local Mapping, Loop Closing and Viewer threads.
    System(const string &strVocFile, 
           const string &strSettingsFile, 
           const eSensor sensor, 
           const bool bUseViewer = true,
           const bool bTrainIntrospectionModel = false,
           const bool bEnableIntrospection = false,
           const bool bSaveVisualizationsToFile = false,
           const bool bCreateIntrospectionDataset = false,
           const bool bSingleThreaded = false,
           const bool bUseBoW = true,
           const string strSaveVisualizationPath = "",
           const string strOutputIntrospectionDatasetPath = "",
           const bool bSilent = false,
           const bool bGuidedBA=false,
           ORBVocabulary* const pVocabulary = NULL);

    // Proccess the given stereo frame. Images must be synchronized and rectified.
    // Input images: RGB (CV_8UC3) or grayscale (CV_8U). RGB is converted to grayscale.
    // Returns the camera pose (empty if tracking fails).
    cv::Mat TrackStereo(const cv::Mat &imLeft, 
                        const cv::Mat &imRight, 
                        const double &timestamp);
    
    // This method is used for training the introspection model using the 
    // ground truth camera pose that is passed as input
    cv::Mat TrackStereo(const cv::Mat &imLeft, 
                        const cv::Mat &imRight, 
                        const double &timestamp,
                        const cv::Mat& cam_pose_gt,
                        const Eigen::Matrix<double, 6, 6>& cam_pose_gt_cov,
                        const bool &pose_cov_available,
                        const std::string& img_name,
                        const bool &gtDepthAvailable = false,
                        const cv::Mat &depthmap = cv::Mat(0,0, CV_32F));
    

    // Process the given rgbd frame. Depthmap must be registered to the RGB frame.
    // Input image: RGB (CV_8UC3) or grayscale (CV_8U). RGB is converted to grayscale.
    // Input depthmap: Float (CV_32F).
    // Returns the camera pose (empty if tracking fails).
    cv::Mat TrackRGBD(const cv::Mat &im, const cv::Mat &depthmap, const double &timestamp);

    // Proccess the given monocular frame
    // Input images: RGB (CV_8UC3) or grayscale (CV_8U). RGB is converted to grayscale.
    // Returns the camera pose (empty if tracking fails).
    cv::Mat TrackMonocular(const cv::Mat &im, const double &timestamp);
    
    // This method is used for training the introspection model using the 
    // ground truth camera pose that is passed as input
    // The ground truth depth map is also an optional argument that could be
    // passed to this function. In order for imDepth to actually be used as 
    // ground truth depth gtDepthAvailable should be set to true. If 
    // gtDepthAvailable is "false" and imDepth is provided, it is assumed
    // that it is a predicted image quality heatmap to be used for scoring
    // the extracted keypoints.
    cv::Mat TrackMonocular(const cv::Mat &im, 
                           const double &timestamp,
                           const cv::Mat& cam_pose_gt,
                           const std::string& img_name,
                           const bool &gtDepthAvailable = false,
                           const cv::Mat &depthmap = cv::Mat(0,0, CV_32F));

    // This stops local mapping thread (map building) and performs only camera tracking.
    void ActivateLocalizationMode();
    // This resumes local mapping thread and performs SLAM again.
    void DeactivateLocalizationMode();

    // Returns true if there have been a big map change (loop closure, global BA)
    // since last call to this function
    bool MapChanged();

    // Reset the system (clear map)
    void Reset();

    // All threads will be requested to finish.
    // It waits until all threads have finished.
    // This function must be called before saving the trajectory.
    void Shutdown();
    
    // A minimal version of shutdown. It does not free all allocated memory
    // as does Shutdown(). The purpose of this method is only for being used
    // in SigInt handlers. For shutting down the system during a running 
    // program use Shutdown to prevent memory leaks.
    void ShutdownMinimal();

    // Save camera trajectory in the TUM RGB-D dataset format.
    // Only for stereo and RGB-D. This method does not work for monocular.
    // Call first Shutdown()
    // See format details at: http://vision.in.tum.de/data/datasets/rgbd-dataset
    void SaveTrajectoryTUM(const string &filename);

    // Save keyframe poses in the TUM RGB-D dataset format.
    // This method works for all sensor input.
    // Call first Shutdown()
    // See format details at: http://vision.in.tum.de/data/datasets/rgbd-dataset
    void SaveKeyFrameTrajectoryTUM(const string &filename);

    // Save camera trajectory in the KITTI dataset format.
    // Only for stereo and RGB-D. This method does not work for monocular.
    // Call first Shutdown()
    // See format details at: http://www.cvlibs.net/datasets/kitti/eval_odometry.php
    void SaveTrajectoryKITTI(const string &filename, 
                    const string &time_filename="Trajectory_KITTI_time.txt");


    // TODO: Save/Load functions
    // SaveMap(const string &filename);
    // LoadMap(const string &filename);

    // Information from most recent processed frame
    // You can call this right after TrackMonocular (or stereo or RGBD)
    int GetTrackingState();
    std::vector<MapPoint*> GetTrackedMapPoints();
    std::vector<cv::KeyPoint> GetTrackedKeyPointsUn();
    
    // Set the list of relative camera pose uncertainty for all frames. This
    // should only be called in training mode and if such information is 
    // available
    void SetRelativeCamPoseUncertainty(
        const  std::unordered_map<std::string, int>* pose_unc_map,
        const vector<Eigen::Vector2f>* rel_cam_poses_uncertainty);
    
    // Returns false if the current tracking state is anything other than OK
    bool GetCurrentCamPose(cv::Mat& cam_pose);

private:
  
    // Input sensor
    eSensor mSensor;

    // ORB vocabulary used for place recognition and feature matching.
    ORBVocabulary* mpVocabulary;

    // KeyFrame database for place recognition (relocalization and loop detection).
    KeyFrameDatabase* mpKeyFrameDatabase;

    // Map structure that stores the pointers to all KeyFrames and MapPoints.
    Map* mpMap;

    // Tracker. It receives a frame and computes the associated camera pose.
    // It also decides when to insert a new keyframe, create some new MapPoints and
    // performs relocalization if tracking fails.
    Tracking* mpTracker;

    // Local Mapper. It manages the local map and performs local bundle adjustment.
    LocalMapping* mpLocalMapper;

    // Loop Closer. It searches loops with every new keyframe. If there is a loop it performs
    // a pose graph optimization and full bundle adjustment (in a new thread) afterwards.
    LoopClosing* mpLoopCloser;

    // The viewer draws the map and the current camera pose. It uses Pangolin.
    Viewer* mpViewer;

    FrameDrawer* mpFrameDrawer;
    MapDrawer* mpMapDrawer;

    // System threads: Local Mapping, Loop Closing, Viewer.
    // The Tracking thread "lives" in the main execution thread that creates the System object.
    std::thread* mptLocalMapping;
    std::thread* mptLoopClosing;
    std::thread* mptViewer;

    // Reset flag
    std::mutex mMutexReset;
    bool mbReset;

    // Change mode flags
    std::mutex mMutexMode;
    bool mbActivateLocalizationMode;
    bool mbDeactivateLocalizationMode;

    // Tracking state
    int mTrackingState;
    std::vector<MapPoint*> mTrackedMapPoints;
    std::vector<cv::KeyPoint> mTrackedKeyPointsUn;
    std::mutex mMutexState;
    
    // In single threaded mode, tracking and local mapping run in the same 
    // thread and loop closing is disabled
    bool mbSingleThreaded;
    
    // Almost no print outs in silent mode. Also the tracked trajectory is not
    // saved to file upon shutdown
    bool mbSilent;
    
    // If set to true and in training mode, camera posese are fixed to 
    // ground truth values. This is to evaluate map points given their 
    // reprojection error
    bool mbGuidedBA;
};

}// namespace ORB_SLAM

#endif // SYSTEM_H
