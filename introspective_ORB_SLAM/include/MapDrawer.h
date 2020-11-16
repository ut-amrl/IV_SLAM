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

#ifndef MAPDRAWER_H
#define MAPDRAWER_H

#include"Map.h"
#include"MapPoint.h"
#include"KeyFrame.h"
#include<pangolin/pangolin.h>
#include<algorithm>

#include<mutex>

DECLARE_bool(map_drawer_visualize_gt_pose);

namespace ORB_SLAM2
{

class MapDrawer
{
public:
    MapDrawer(Map* pMap, const string &strSettingPath);

    Map* mpMap;

    void DrawMapPoints();
    void DrawKeyFrames(const bool bDrawKF, const bool bDrawGraph);
    void DrawCurrentCamera(pangolin::OpenGlMatrix &Twc);
    void DrawCurrentCameraGT();
    void SetCurrentCameraPose(const cv::Mat &Tcw);
    void SetCurrentCameraPose(const cv::Mat &Tcw, 
                              const std::string& strFrameName);
    void SetCurrentCameraPosewithGT(const cv::Mat &Tcw,
                                    const cv::Mat &Twc_gt, 
                              const std::string& strFrameName);
    void SetReferenceKeyFrame(KeyFrame *pKF);
    void GetCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix &M);
    void ConverT2OpenGLCameraMatrix(const cv::Mat &Twc,
                                    pangolin::OpenGlMatrix &M);
    
    // Returns the transformation from src_frame to dest_frame (takes a point 
    // from src_frame to dest_frame)
    cv::Mat CalculateRelativeTransform(const cv::Mat& dest_frame_pose,
                                       const cv::Mat& src_frame_pose);
    cv::Mat CalculateInverseTransform(const cv::Mat& transform);
    
    std::string mstrFrameName;

private:

    const int mGTCameraPoseHistory = 10;

    float mKeyFrameSize;
    float mKeyFrameLineWidth;
    float mGraphLineWidth;
    float mPointSize;
    float mCameraSize;
    float mCameraLineWidth;
    
    // Estimated transformation from world frame to the current camera frame
    cv::Mat mCameraPose;
    
    // Ground truth transformation from camera frame to the world frame
    cv::Mat mTwc_gt;
    bool mbGTPoseAvailable = false;
    
    // The ground truth pose of the oldest keyframe in the ground truth pose 
    // visualization queue (This queue is of length mGTCameraPoseHistory)
    cv::Mat mTwc_gt_vis_init;
    // The estimated pose of the oldest keyframe in the visualization queue
    cv::Mat mTwc_vis_init;

    std::mutex mMutexCamera;
};

} //namespace ORB_SLAM

#endif // MAPDRAWER_H
