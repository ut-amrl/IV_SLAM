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

#include "FrameDrawer.h"
#include "Tracking.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include<mutex>

namespace ORB_SLAM2
{

FrameDrawer::FrameDrawer(Map* pMap):mpMap(pMap)
{
    mState=Tracking::SYSTEM_NOT_READY;
    mIm = cv::Mat(480,640,CV_8UC3, cv::Scalar(0,0,0));
}

cv::Mat FrameDrawer::DrawFrame(bool save_to_file)
{
    cv::Mat im;
    vector<cv::KeyPoint> vIniKeys; // Initialization: KeyPoints in reference frame
    vector<int> vMatches; // Initialization: correspondeces with reference keypoints
    vector<cv::KeyPoint> vCurrentKeys; // KeyPoints in current frame
    vector<float> vCurrentKeysQual; // Quality score of KeyPoints in current 
                                    // frame
    vector<bool> vbVO, vbMap; // Tracked MapPoints in current frame
    int state; // Tracking state

    //Copy variables within scoped mutex
    {
        unique_lock<mutex> lock(mMutex);
        state=mState;
        if(mState==Tracking::SYSTEM_NOT_READY)
            mState=Tracking::NO_IMAGES_YET;

        mIm.copyTo(im);

        if(mState==Tracking::NOT_INITIALIZED)
        {
            vCurrentKeys = mvCurrentKeys;
            vCurrentKeysQual = mvCurrentKeysQualScore;
            vIniKeys = mvIniKeys;
            vMatches = mvIniMatches;
        }
        else if(mState==Tracking::OK)
        {
            vCurrentKeys = mvCurrentKeys;
            vCurrentKeysQual = mvCurrentKeysQualScore;
            vbVO = mvbVO;
            vbMap = mvbMap;
        }
        else if(mState==Tracking::LOST)
        {
            vCurrentKeys = mvCurrentKeys;
            vCurrentKeysQual = mvCurrentKeysQualScore;
        }
    } // destroy scoped mutex -> release mutex

    if(im.channels()<3) //this should be always true
        cvtColor(im,im,CV_GRAY2BGR);

    //Draw
    if(state==Tracking::NOT_INITIALIZED) //INITIALIZING
    {
        for(unsigned int i=0; i<vMatches.size(); i++)
        {
            if(vMatches[i]>=0)
            {
                cv::line(im,vIniKeys[i].pt,vCurrentKeys[vMatches[i]].pt,
                        cv::Scalar(0,255,0));
            }
        }        
    }
    else if(state==Tracking::OK) //TRACKING
    {
        mnTracked=0;
        mnTrackedVO=0;
        const float r = 5; // def: 5
        const int thickness = 1; // def: 1
        const int n = vCurrentKeys.size();
        for(int i=0;i<n;i++)
        {
            if(vbVO[i] || vbMap[i])
            {
                cv::Point2f pt1,pt2;
                pt1.x=vCurrentKeys[i].pt.x-r;
                pt1.y=vCurrentKeys[i].pt.y-r;
                pt2.x=vCurrentKeys[i].pt.x+r;
                pt2.y=vCurrentKeys[i].pt.y+r;
                
                // Green at high quality, red at low quality
                cv::Scalar color = cv::Scalar(0.0,
                                              255 * vCurrentKeysQual[i],
                                            255 - 255 * vCurrentKeysQual[i]);

                // This is a match to a MapPoint in the map
                if(vbMap[i])
                {
                    cv::rectangle(im,pt1,pt2,color, thickness);
                    cv::circle(im,vCurrentKeys[i].pt,2,color,-1);
                    mnTracked++;
                }
                else // This is match to a "visual odometry" MapPoint created in the last frame
                {
                    cv::rectangle(im,pt1,pt2,cv::Scalar(255,0,0), thickness);
                    cv::circle(im,vCurrentKeys[i].pt,2,cv::Scalar(255,0,0),-1);
                    mnTrackedVO++;
                }
            }
        }
    }

    cv::Mat imWithInfo;
    DrawTextInfo(im,state, imWithInfo);
    
    if (save_to_file) {
      SaveToFile(imWithInfo);
    }
      
    return imWithInfo;
}


void FrameDrawer::SaveToFile(const cv::Mat &im) {
  static bool first_call = true;
  
  if (!mbVisualizationPathSet) {
    return;
  }

  if (first_call) {
    RemoveDirectory(mstrSaveVisualizationPath + "/frame_drawer/");
    CreateDirectory(mstrSaveVisualizationPath + "/frame_drawer/");
  }

  string img_name_truncated = mstrFrameName.substr(0, mstrFrameName.length()-4);
  string file_path = mstrSaveVisualizationPath + "/frame_drawer/" + 
                      img_name_truncated + ".jpg";
  cv::imwrite(file_path, im);
  
  first_call = false;
}


void FrameDrawer::DrawTextInfo(cv::Mat &im, int nState, cv::Mat &imText)
{
    stringstream s;
    if(nState==Tracking::NO_IMAGES_YET)
        s << " WAITING FOR IMAGES";
    else if(nState==Tracking::NOT_INITIALIZED)
        s << " TRYING TO INITIALIZE ";
    else if(nState==Tracking::OK)
    {
        if(!mbOnlyTracking)
            s << "SLAM MODE |  ";
        else
            s << "LOCALIZATION | ";
        int nKFs = mpMap->KeyFramesInMap();
        int nMPs = mpMap->MapPointsInMap();
        s << "KFs: " << nKFs << ", MPs: " << nMPs << ", Matches: " << mnTracked;
        if(mnTrackedVO>0)
            s << ", + VO matches: " << mnTrackedVO;
    }
    else if(nState==Tracking::LOST)
    {
        s << " TRACK LOST. TRYING TO RELOCALIZE ";
    }
    else if(nState==Tracking::SYSTEM_NOT_READY)
    {
        s << " LOADING ORB VOCABULARY. PLEASE WAIT...";
    }

    int baseline=0;
    cv::Size textSize = cv::getTextSize(s.str(),cv::FONT_HERSHEY_PLAIN,1,1,&baseline);

    imText = cv::Mat(im.rows+textSize.height+10,im.cols,im.type());
    im.copyTo(imText.rowRange(0,im.rows).colRange(0,im.cols));
    imText.rowRange(im.rows,imText.rows) = cv::Mat::zeros(textSize.height+10,im.cols,im.type());
    cv::putText(imText,s.str(),cv::Point(5,imText.rows-5),cv::FONT_HERSHEY_PLAIN,1,cv::Scalar(255,255,255),1,8);

    // Annotate restart events such that they stay for a few frames
    if (mpMap->KeyFramesInMap()<=15 && mLastFrameId > 300 && true) { // < 15
      string txt = "Restarted";
      double font_scale = 4.0;
      int thickness = 4;
      
      cv::Size txt_size = 
              cv::getTextSize(txt,cv::FONT_HERSHEY_PLAIN, 
                             font_scale,thickness,&baseline);
          
      cv::putText(imText,txt,
                  cv::Point(5, 20 + txt_size.height),
                  cv::FONT_HERSHEY_PLAIN,font_scale,
                  cv::Scalar(0,0,255), thickness,8);
    }
}

void FrameDrawer::Update(Tracking *pTracker)
{ 
    static bool first_call = true;
    
    unique_lock<mutex> lock(mMutex);
    pTracker->mImGray.copyTo(mIm);
    mvCurrentKeys=pTracker->mCurrentFrame.mvKeys;  
    N = mvCurrentKeys.size();
    mvbVO = vector<bool>(N,false);
    mvbMap = vector<bool>(N,false);
    mbOnlyTracking = pTracker->mbOnlyTracking;
    mLastFrameId = pTracker->mCurrentFrame.mnId;
    
    if(pTracker->mbUnsupervisedLearning) {
      mvCurrentKeysQualScore = pTracker->mCurrentFrame.mvKeyQualScoreTrain;
    } else {
      mvCurrentKeysQualScore = pTracker->mCurrentFrame.mvKeyQualScore;
    }


    if(pTracker->mLastProcessedState==Tracking::NOT_INITIALIZED)
    {
        mvIniKeys=pTracker->mInitialFrame.mvKeys;
        mvIniMatches=pTracker->mvIniMatches;
    }
    else if(pTracker->mLastProcessedState==Tracking::OK)
    {
        for(int i=0;i<N;i++)
        {
          if (!pTracker->mbUnsupervisedLearning) {
            MapPoint* pMP = pTracker->mCurrentFrame.mvpMapPoints[i];
            if(pMP)
            {
                if(!pTracker->mCurrentFrame.mvbOutlier[i])
                {
                    if(pMP->Observations()>0)
                        mvbMap[i]=true;
                    else
                        mvbVO[i]=true;
                }
            }
          } else {
            // Visualize anything that
            // is suggested by Frame.ComputeKeyPtQualScores(), which usually
            // also includes map keypoints that have been pruend as outliers 
            if (pTracker->mCurrentFrame.mvChi2Dof[i] > 0) {
              mvbMap[i] = true;
            } else {
              mvbMap[i] = false;
            }
          }
        }
    }
    mState=static_cast<int>(pTracker->mLastProcessedState);
    mstrFrameName = pTracker->mCurrentFrame.mstrLeftImgName;
    
    if (first_call) {
      mstrSaveVisualizationPath = pTracker->mvSaveVisualizationPath;
      mbVisualizationPathSet = true;
    }
    
    first_call = false;
}

} //namespace ORB_SLAM
