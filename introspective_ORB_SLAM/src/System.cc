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



#include "System.h"
#include "Converter.h"
#include <thread>
#include <pangolin/pangolin.h>
#include <iomanip>

namespace ORB_SLAM2
{

System::System(const string &strVocFile,
               const string &strSettingsFile, 
               const eSensor sensor,
               const bool bUseViewer,
               const bool bTrainIntrospectionModel,
               const bool bEnableIntrospection,
               const bool bSaveVisualizationsToFile,
               const bool bCreateIntrospectionDataset,
               const bool bSingleThreaded,
               const bool bUseBoW,
               const string strSaveVisualizationPath,
               const string strOutputIntrospectionDatasetPath,
               const bool bSilent,
               const bool bGuidedBA,
               ORBVocabulary* const pVocabulary): 
                      mSensor(sensor), 
                      mpViewer(static_cast<Viewer*>(NULL)), 
                      mbReset(false),mbActivateLocalizationMode(false),
                      mbDeactivateLocalizationMode(false),
                      mbSingleThreaded(bSingleThreaded),
                      mbSilent(bSilent),
                      mbGuidedBA(bGuidedBA)
{
    if(!bSilent) {
      // Output welcome message
      cout << endl <<
      "ORB-SLAM2 Copyright (C) 2014-2016 Raul Mur-Artal, University of "
      "Zaragoza." << endl <<
      "This program comes with ABSOLUTELY NO WARRANTY;" << endl  <<
      "This is free software, and you are welcome to redistribute it" << endl <<
      "under certain conditions. See LICENSE.txt." << endl << endl;

      cout << "Input sensor was set to: ";

      if(mSensor==MONOCULAR)
          cout << "Monocular" << endl;
      else if(mSensor==STEREO)
          cout << "Stereo" << endl;
      else if(mSensor==RGBD)
          cout << "RGB-D" << endl;
    }

    //Check settings file
    if(!bSilent) {
      cv::FileStorage fsSettings(strSettingsFile.c_str(), 
                                 cv::FileStorage::READ);
      if(!fsSettings.isOpened()) {
        cerr << "Failed to open settings file at: " << strSettingsFile << endl;
        exit(-1);
      } else {
        fsSettings.release();
      }
    }

    KeyFrame::nNextId=0;
    Frame::nNextId=0;

    // If bUseBoW is set to false, the functionalities that rely on bag of 
    // words will be turned off. This includes loop closure and tracking 
    // with reference frame where feature matching is done with searching
    // extracted bag of words for each frame.
    if (bUseBoW) {
      //Load ORB Vocabulary
      mpVocabulary = new ORBVocabulary();
      
      if(pVocabulary) {
        // IF a vocabulary is passed, use that instead of loading from file
        *mpVocabulary = *pVocabulary;
      } else {
        cout << endl << "Loading ORB Vocabulary. This could take a while..." << 
              endl;
        bool bVocLoad = mpVocabulary->loadFromTextFile(strVocFile);
        if(!bVocLoad)
        {
            cerr << "Wrong path to vocabulary. " << endl;
            cerr << "Falied to open at: " << strVocFile << endl;
            exit(-1);
        }
        cout << "Vocabulary loaded!" << endl << endl;
      }
    } else {
      mpVocabulary = NULL;
    }
    


    //Create KeyFrame Database
    if (bUseBoW) {
      mpKeyFrameDatabase = new KeyFrameDatabase(*mpVocabulary);
    } else {
      mpKeyFrameDatabase = NULL;
    }
      

    //Create the Map
    mpMap = new Map();

    //Create Drawers. These are used by the Viewer
    if(bUseViewer) {
      mpFrameDrawer = new FrameDrawer(mpMap);
    } else {
      mpFrameDrawer = NULL;
    }
    mpMapDrawer = new MapDrawer(mpMap, strSettingsFile);

    //Initialize the Tracking thread
    //(it will live in the main thread of execution, the one that called this constructor)
    mpTracker = new Tracking(this, mpVocabulary, mpFrameDrawer, mpMapDrawer,
                             mpMap, mpKeyFrameDatabase, strSettingsFile, 
                             mSensor, bSingleThreaded, bSilent, bGuidedBA);

    mpTracker->mbIntrospectionOn = bEnableIntrospection;
    mpTracker->mbTrainingMode = bTrainIntrospectionModel;
    mpTracker->mbSaveVisualizationsToFile = bSaveVisualizationsToFile;
    mpTracker->mbCreateIntrospectionDataset = bCreateIntrospectionDataset;
    mpTracker->mvSaveVisualizationPath = strSaveVisualizationPath;
    mpTracker->mvOutputIntrospectionDatasetPath = 
                                      strOutputIntrospectionDatasetPath;
                                      
    // Modify strOutputIntrospectionDatasetPath to create the path for the
    // comprehensive dataset that includes all images and not only those
    // selected for training
    if (bCreateIntrospectionDataset) {                                     
      std::string strModifiedDataPath = strOutputIntrospectionDatasetPath;
      size_t pos0 = strModifiedDataPath.length() - 6;
      std::string session_suffix = strModifiedDataPath.substr(pos0, 6);
      strModifiedDataPath = strModifiedDataPath.substr(0, pos0) + 
                            "Comprehensive/" + session_suffix;
      cout << "Comprehensive Dataset Path: " << endl;
      cout << strModifiedDataPath << endl;
      mpTracker->mvOutputIntrospectionDatasetFullPath = 
                                        strModifiedDataPath;
    }
                                                                
    
    
    //Initialize the Local Mapping thread and launch
    mpLocalMapper = new LocalMapping(mpMap, mSensor==MONOCULAR);
    
    // In single threaded mode, local mapping optimizations are called in 
    // the tracking thread
    if (!bSingleThreaded) {
      mptLocalMapping = new thread(&ORB_SLAM2::LocalMapping::Run,mpLocalMapper);
    }
      
    //Initialize the Loop Closing thread and launch
    if (!bSingleThreaded) {
      mpLoopCloser = new LoopClosing(mpMap, 
                                    mpKeyFrameDatabase, 
                                    mpVocabulary, 
                                    mSensor!=MONOCULAR);
      mptLoopClosing = new thread(&ORB_SLAM2::LoopClosing::Run,
                                  mpLoopCloser);
    } else {
      mpLoopCloser = NULL;
      mptLoopClosing = NULL;
    }

    //Initialize the Viewer thread and launch
    if(bUseViewer)
    {
        mpViewer = new Viewer(this, mpFrameDrawer,mpMapDrawer,mpTracker,strSettingsFile);
        mptViewer = new thread(&Viewer::Run, mpViewer);
        mpTracker->SetViewer(mpViewer);
    }

    //Set pointers between threads
    mpTracker->SetLocalMapper(mpLocalMapper);
    mpTracker->SetLoopClosing(mpLoopCloser);

    mpLocalMapper->SetTracker(mpTracker);
    mpLocalMapper->SetLoopCloser(mpLoopCloser);

    if (mpLoopCloser) {
      mpLoopCloser->SetTracker(mpTracker);
      mpLoopCloser->SetLocalMapper(mpLocalMapper);
    }
   
    
}

cv::Mat System::TrackStereo(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timestamp)
{
    if(mSensor!=STEREO)
    {
        cerr << "ERROR: you called TrackStereo but input sensor was not set to STEREO." << endl;
        exit(-1);
    }   

    // Check mode change
    {
        unique_lock<mutex> lock(mMutexMode);
        if(mbActivateLocalizationMode)
        {
            mpLocalMapper->RequestStop();

            // Wait until Local Mapping has effectively stopped
            while(!mpLocalMapper->isStopped())
            {
                usleep(1000);
            }

            mpTracker->InformOnlyTracking(true);
            mbActivateLocalizationMode = false;
        }
        if(mbDeactivateLocalizationMode)
        {
            mpTracker->InformOnlyTracking(false);
            mpLocalMapper->Release();
            mbDeactivateLocalizationMode = false;
        }
    }

    // Check reset
    {
    unique_lock<mutex> lock(mMutexReset);
    if(mbReset)
    {
        mpTracker->Reset();
        mbReset = false;
    }
    }

    cv::Mat Tcw = mpTracker->GrabImageStereo(imLeft,imRight,timestamp);

    unique_lock<mutex> lock2(mMutexState);
    mTrackingState = mpTracker->mState;
    mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
    mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;
    return Tcw;
}


cv::Mat System::TrackStereo(const cv::Mat &imLeft, 
                            const cv::Mat &imRight, 
                            const double &timestamp,
                            const cv::Mat& cam_pose_gt,
                          const Eigen::Matrix<double, 6, 6>& cam_pose_gt_cov,
                          const bool &pose_cov_available,
                            const std::string& img_name,
                            const bool &gtDepthAvailable,
                            const cv::Mat &depthmap) {
    if(mSensor!=STEREO)
    {
        cerr << "ERROR: you called TrackStereo but input sensor was "
                 "not set to STEREO." << endl;
        exit(-1);
    }   

    // Check mode change
    {
        unique_lock<mutex> lock(mMutexMode);
        if(mbActivateLocalizationMode)
        {
            mpLocalMapper->RequestStop();

            // Wait until Local Mapping has effectively stopped
            while(!mpLocalMapper->isStopped())
            {
                usleep(1000);
            }

            mpTracker->InformOnlyTracking(true);
            mbActivateLocalizationMode = false;
        }
        if(mbDeactivateLocalizationMode)
        {
            mpTracker->InformOnlyTracking(false);
            mpLocalMapper->Release();
            mbDeactivateLocalizationMode = false;
        }
    }

    // Check reset
    {
    unique_lock<mutex> lock(mMutexReset);
    if(mbReset)
    {
        mpTracker->Reset();
        mbReset = false;
    }
    }

    cv::Mat Tcw = mpTracker->GrabImageStereo(imLeft,
                                             imRight,
                                             timestamp,
                                             cam_pose_gt,
                                             cam_pose_gt_cov,
                                             pose_cov_available,
                                             img_name,
                                             gtDepthAvailable,
                                             depthmap);

    unique_lock<mutex> lock2(mMutexState);
    mTrackingState = mpTracker->mState;
    mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
    mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;
    return Tcw;
}

cv::Mat System::TrackRGBD(const cv::Mat &im, const cv::Mat &depthmap, const double &timestamp)
{
    if(mSensor!=RGBD)
    {
        cerr << "ERROR: you called TrackRGBD but input sensor was not set to RGBD." << endl;
        exit(-1);
    }    

    // Check mode change
    {
        unique_lock<mutex> lock(mMutexMode);
        if(mbActivateLocalizationMode)
        {
            mpLocalMapper->RequestStop();

            // Wait until Local Mapping has effectively stopped
            while(!mpLocalMapper->isStopped())
            {
                usleep(1000);
            }

            mpTracker->InformOnlyTracking(true);
            mbActivateLocalizationMode = false;
        }
        if(mbDeactivateLocalizationMode)
        {
            mpTracker->InformOnlyTracking(false);
            mpLocalMapper->Release();
            mbDeactivateLocalizationMode = false;
        }
    }

    // Check reset
    {
    unique_lock<mutex> lock(mMutexReset);
    if(mbReset)
    {
        mpTracker->Reset();
        mbReset = false;
    }
    }

    cv::Mat Tcw = mpTracker->GrabImageRGBD(im,depthmap,timestamp);

    unique_lock<mutex> lock2(mMutexState);
    mTrackingState = mpTracker->mState;
    mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
    mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;
    return Tcw;
}

cv::Mat System::TrackMonocular(const cv::Mat &im, const double &timestamp)
{
    if(mSensor!=MONOCULAR)
    {
        cerr << "ERROR: you called TrackMonocular but input sensor was not set to Monocular." << endl;
        exit(-1);
    }

    // Check mode change
    {
        unique_lock<mutex> lock(mMutexMode);
        if(mbActivateLocalizationMode)
        {
            mpLocalMapper->RequestStop();

            // Wait until Local Mapping has effectively stopped
            while(!mpLocalMapper->isStopped())
            {
                usleep(1000);
            }

            mpTracker->InformOnlyTracking(true);
            mbActivateLocalizationMode = false;
        }
        if(mbDeactivateLocalizationMode)
        {
            mpTracker->InformOnlyTracking(false);
            mpLocalMapper->Release();
            mbDeactivateLocalizationMode = false;
        }
    }

    // Check reset
    {
    unique_lock<mutex> lock(mMutexReset);
    if(mbReset)
    {
        mpTracker->Reset();
        mbReset = false;
    }
    }

    cv::Mat Tcw = mpTracker->GrabImageMonocular(im,timestamp);

    unique_lock<mutex> lock2(mMutexState);
    mTrackingState = mpTracker->mState;
    mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
    mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;

    return Tcw;
}

cv::Mat System::TrackMonocular(const cv::Mat &im, 
                           const double &timestamp,
                           const cv::Mat& cam_pose_gt,
                           const std::string& img_name,
                           const bool &gtDepthAvailable,
                           const cv::Mat &depthmap) {    
    if(mSensor!=MONOCULAR) {
        cerr << "ERROR: you called TrackMonocular but input sensor was "
        "not set to Monocular." << endl;
        exit(-1);
    }

    // Check mode change
    {
        unique_lock<mutex> lock(mMutexMode);
        if(mbActivateLocalizationMode)
        {
            mpLocalMapper->RequestStop();

            // Wait until Local Mapping has effectively stopped
            while(!mpLocalMapper->isStopped())
            {
                usleep(1000);
            }

            mpTracker->InformOnlyTracking(true);
            mbActivateLocalizationMode = false;
        }
        if(mbDeactivateLocalizationMode)
        {
            mpTracker->InformOnlyTracking(false);
            mpLocalMapper->Release();
            mbDeactivateLocalizationMode = false;
        }
    }

    // Check reset
    {
    unique_lock<mutex> lock(mMutexReset);
    if(mbReset)
    {
        mpTracker->Reset();
        mbReset = false;
    }
    }

    cv::Mat Tcw = mpTracker->GrabImageMonocular(im,
                                                timestamp, 
                                                cam_pose_gt,
                                                img_name,
                                                gtDepthAvailable,
                                                depthmap);

    unique_lock<mutex> lock2(mMutexState);
    mTrackingState = mpTracker->mState;
    mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
    mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;

    return Tcw;
}

void System::ActivateLocalizationMode()
{
    unique_lock<mutex> lock(mMutexMode);
    mbActivateLocalizationMode = true;
}

void System::DeactivateLocalizationMode()
{
    unique_lock<mutex> lock(mMutexMode);
    mbDeactivateLocalizationMode = true;
}

bool System::MapChanged()
{
    static int n=0;
    int curn = mpMap->GetLastBigChangeIdx();
    if(n<curn)
    {
        n=curn;
        return true;
    }
    else
        return false;
}

void System::Reset()
{
    unique_lock<mutex> lock(mMutexReset);
    mbReset = true;
}

void System::Shutdown()
{  
    mpTracker->SaveIntrospectionDataset();
    if(!mbSilent) {
      mpTracker->SaveTrackingResults(false);
    }
    mpTracker->Release();
   
    
    mpLocalMapper->RequestFinish();
    mpLocalMapper->Release();
   
    
    if (mpLoopCloser) {
      mpLoopCloser->RequestFinish();
    }
    if(mpViewer)
    {
        mpViewer->RequestFinish();
        while(!mpViewer->isFinished())
            usleep(5000);
    }

    // Wait until all thread have effectively stopped
    if (!mbSingleThreaded)
      while(!mpLocalMapper->isFinished() || 
            !mpLoopCloser->isFinished() || 
            mpLoopCloser->isRunningGBA()) {
          usleep(5000);
      }

      
    if(mpViewer) {
//         pangolin::BindToContext("ORB-SLAM2: Map Viewer");
        pangolin::DestroyWindow("ORB-SLAM2: Map Viewer");
        
        delete mpViewer;
        mpViewer = static_cast<Viewer*>(NULL);
    }
    
    mpMap->clear();
   
    delete mpMap;
    delete mpMapDrawer;
    delete mpTracker;
    delete mpLocalMapper;

    if (mpFrameDrawer) {
      delete mpFrameDrawer;
    }
    
    if (mpLoopCloser){
      delete mpLoopCloser;
    }
    
    if (mpKeyFrameDatabase) {
      mpKeyFrameDatabase->clear();
      delete mpKeyFrameDatabase;
    }
    
    if (mpVocabulary){
      delete mpVocabulary;
    }
}

void System::ShutdownMinimal()
{  
    mpTracker->SaveIntrospectionDataset();
    if(!mbSilent) {
      mpTracker->SaveTrackingResults(false);
    }
    
    mpLocalMapper->RequestFinish();
    mpLocalMapper->Release();
   
    
    if (mpLoopCloser) {
      mpLoopCloser->RequestFinish();
    }
    if(mpViewer)
    {
        mpViewer->RequestFinish();
        while(!mpViewer->isFinished())
            usleep(5000);
    }

    // Wait until all thread have effectively stopped
    if (!mbSingleThreaded)
      while(!mpLocalMapper->isFinished() || 
            !mpLoopCloser->isFinished() || 
            mpLoopCloser->isRunningGBA()) {
          usleep(5000);
      }

      
    if(mpViewer) {
//         pangolin::BindToContext("ORB-SLAM2: Map Viewer");
        pangolin::DestroyWindow("ORB-SLAM2: Map Viewer");
        
        delete mpViewer;
        mpViewer = static_cast<Viewer*>(NULL);
    }
    
    mpMap->clear();
   
    delete mpMap;
    delete mpMapDrawer;
    delete mpTracker;
    delete mpLocalMapper;

    if (mpFrameDrawer) {
      delete mpFrameDrawer;
    }
    
    if (mpLoopCloser){
      delete mpLoopCloser;
    }
    
    
    if (mpKeyFrameDatabase) {
      mpKeyFrameDatabase->clear();
      delete mpKeyFrameDatabase;
    }
    
    if (mpVocabulary){
      delete mpVocabulary;
    }
}

void System::SaveTrajectoryTUM(const string &filename)
{
    cout << endl << "Saving camera trajectory to " << filename << " ..." << endl;
    if(mSensor==MONOCULAR)
    {
        cerr << "ERROR: SaveTrajectoryTUM cannot be used for monocular." << endl;
        return;
    }

    vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
    sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    cv::Mat Two = vpKFs[0]->GetPoseInverse();

    ofstream f;
    f.open(filename.c_str());
    f << fixed;

    // Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
    // We need to get first the keyframe pose and then concatenate the relative transformation.
    // Frames not localized (tracking failure) are not saved.

    // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
    // which is true when tracking failed (lbL).
    list<ORB_SLAM2::KeyFrame*>::iterator lRit = mpTracker->mlpReferences.begin();
    list<double>::iterator lT = mpTracker->mlFrameTimes.begin();
    list<bool>::iterator lbL = mpTracker->mlbLost.begin();
    for(list<cv::Mat>::iterator lit=mpTracker->mlRelativeFramePoses.begin(),
        lend=mpTracker->mlRelativeFramePoses.end();lit!=lend;lit++, lRit++, lT++, lbL++)
    {
        if(*lbL)
            continue;

        KeyFrame* pKF = *lRit;

        cv::Mat Trw = cv::Mat::eye(4,4,CV_32F);

        // If the reference keyframe was culled, traverse the spanning tree to get a suitable keyframe.
        while(pKF->isBad())
        {
            Trw = Trw*pKF->mTcp;
            pKF = pKF->GetParent();
        }

        Trw = Trw*pKF->GetPose()*Two;

        cv::Mat Tcw = (*lit)*Trw;
        cv::Mat Rwc = Tcw.rowRange(0,3).colRange(0,3).t();
        cv::Mat twc = -Rwc*Tcw.rowRange(0,3).col(3);

        vector<float> q = Converter::toQuaternion(Rwc);

        f << setprecision(6) << *lT << " " <<  setprecision(9) << twc.at<float>(0) << " " << twc.at<float>(1) << " " << twc.at<float>(2) << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;
    }
    f.close();
    cout << endl << "trajectory saved!" << endl;
}


void System::SaveKeyFrameTrajectoryTUM(const string &filename)
{
    cout << endl << "Saving keyframe trajectory to " << filename << " ..." << endl;

    vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
    sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    //cv::Mat Two = vpKFs[0]->GetPoseInverse();

    ofstream f;
    f.open(filename.c_str());
    f << fixed;

    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKF = vpKFs[i];

       // pKF->SetPose(pKF->GetPose()*Two);

        if(pKF->isBad())
            continue;

        cv::Mat R = pKF->GetRotation().t();
        vector<float> q = Converter::toQuaternion(R);
        cv::Mat t = pKF->GetCameraCenter();
        f << setprecision(6) << pKF->mTimeStamp << setprecision(7) << " " << t.at<float>(0) << " " << t.at<float>(1) << " " << t.at<float>(2)
          << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;

    }

    f.close();
    cout << endl << "trajectory saved!" << endl;
}

void System::SaveTrajectoryKITTI(const string &filename,
                                 const string &time_filename)
{
    cout << endl << "Saving camera trajectory to " << filename << " ..." << endl;
    if(mSensor==MONOCULAR)
    {
        cerr << "ERROR: SaveTrajectoryKITTI cannot be used for monocular." << endl;
        return;
    }

    vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
    sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    cv::Mat Two = vpKFs[0]->GetPoseInverse();

    ofstream f;
    f.open(filename.c_str());
    f << fixed;

    ofstream f_time;
    f_time.open(time_filename.c_str());

    // Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
    // We need to get first the keyframe pose and then concatenate the relative transformation.
    // Frames not localized (tracking failure) are not saved.

    // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
    // which is true when tracking failed (lbL).
    list<ORB_SLAM2::KeyFrame*>::iterator lRit = mpTracker->mlpReferences.begin();
    list<double>::iterator lT = mpTracker->mlFrameTimes.begin();
    for(list<cv::Mat>::iterator lit=mpTracker->mlRelativeFramePoses.begin(), lend=mpTracker->mlRelativeFramePoses.end();lit!=lend;lit++, lRit++, lT++)
    {
        ORB_SLAM2::KeyFrame* pKF = *lRit;

        cv::Mat Trw = cv::Mat::eye(4,4,CV_32F);

        while(pKF->isBad())
        {
          //  cout << "bad parent" << endl;
            Trw = Trw*pKF->mTcp;
            pKF = pKF->GetParent();
        }

        Trw = Trw*pKF->GetPose()*Two;

        cv::Mat Tcw = (*lit)*Trw;
        cv::Mat Rwc = Tcw.rowRange(0,3).colRange(0,3).t();
        cv::Mat twc = -Rwc*Tcw.rowRange(0,3).col(3);

        f << setprecision(9) << Rwc.at<float>(0,0) << " " << Rwc.at<float>(0,1)  << " " << Rwc.at<float>(0,2) << " "  << twc.at<float>(0) << " " <<
             Rwc.at<float>(1,0) << " " << Rwc.at<float>(1,1)  << " " << Rwc.at<float>(1,2) << " "  << twc.at<float>(1) << " " <<
             Rwc.at<float>(2,0) << " " << Rwc.at<float>(2,1)  << " " << Rwc.at<float>(2,2) << " "  << twc.at<float>(2) << endl;


        f_time << setprecision(15) << *lT << endl;
    }
    f.close();
    f_time.close();
    cout << endl << "trajectory saved!" << endl;
}

int System::GetTrackingState()
{
    unique_lock<mutex> lock(mMutexState);
    return mTrackingState;
}

void System::SetRelativeCamPoseUncertainty(
      const  std::unordered_map<std::string, int>* pose_unc_map,
      const vector<Eigen::Vector2f>* rel_cam_poses_uncertainty) {
  mpTracker->SetRelativeCamPoseUncertainty(pose_unc_map, 
                                           rel_cam_poses_uncertainty);
}

bool System::GetCurrentCamPose(cv::Mat& cam_pose) {
  unique_lock<mutex> lock(mMutexState);
  
  if (mTrackingState != 2) {
    return false;
  } else {
    cam_pose = mpTracker->mCurrentFramePose;
    return true;
  }
}

vector<MapPoint*> System::GetTrackedMapPoints()
{
    unique_lock<mutex> lock(mMutexState);
    return mTrackedMapPoints;
}

vector<cv::KeyPoint> System::GetTrackedKeyPointsUn()
{
    unique_lock<mutex> lock(mMutexState);
    return mTrackedKeyPointsUn;
}

} //namespace ORB_SLAM
