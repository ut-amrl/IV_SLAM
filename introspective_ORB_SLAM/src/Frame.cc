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

#include "Frame.h"
#include "Converter.h"
#include "ORBmatcher.h"
#include <thread>
#include <boost/math/distributions.hpp>
#include <glog/logging.h>
#include <gflags/gflags.h>

DEFINE_double(ivslam_keypt_qual_chi2_prob_thresh, 0.99, 
             "The maximum probablity "
             "threshold for the normalized reprojection errors (chi2 dist). "
             "This is used to generate the keypoint quality scores from "
             "reprojection erros when in unsupervised learning mode.");

namespace ORB_SLAM2
{

long unsigned int Frame::nNextId=0;
bool Frame::mbInitialComputations=true;
float Frame::cx, Frame::cy, Frame::fx, Frame::fy, Frame::invfx, Frame::invfy;
float Frame::mnMinX, Frame::mnMinY, Frame::mnMaxX, Frame::mnMaxY;
float Frame::mfGridElementWidthInv, Frame::mfGridElementHeightInv;

Frame::Frame()
{}

//Copy Constructor
Frame::Frame(const Frame &frame)
    :mpORBvocabulary(frame.mpORBvocabulary), mpORBextractorLeft(frame.mpORBextractorLeft), mpORBextractorRight(frame.mpORBextractorRight),
     mTimeStamp(frame.mTimeStamp), mK(frame.mK.clone()), mDistCoef(frame.mDistCoef.clone()),
     mbf(frame.mbf), mb(frame.mb), mThDepth(frame.mThDepth), N(frame.N), mvKeys(frame.mvKeys),
     mvKeysRight(frame.mvKeysRight), mvKeysUn(frame.mvKeysUn),
     mvKeysGTDepth(frame.mvKeysGTDepth),
     mvKeyQualScore(frame.mvKeyQualScore), mvuRight(frame.mvuRight),
     mvDepth(frame.mvDepth), mBowVec(frame.mBowVec), mFeatVec(frame.mFeatVec),
     mDescriptors(frame.mDescriptors.clone()), mDescriptorsRight(frame.mDescriptorsRight.clone()),
     mvpMapPoints(frame.mvpMapPoints), mvbOutlier(frame.mvbOutlier), mnId(frame.mnId),
     mstrLeftImgName(frame.mstrLeftImgName),
     mpReferenceKF(frame.mpReferenceKF), mnScaleLevels(frame.mnScaleLevels),
     mfScaleFactor(frame.mfScaleFactor), mfLogScaleFactor(frame.mfLogScaleFactor),
     mvScaleFactors(frame.mvScaleFactors), mvInvScaleFactors(frame.mvInvScaleFactors),
     mvLevelSigma2(frame.mvLevelSigma2), 
     mvInvLevelSigma2(frame.mvInvLevelSigma2),
     mvKeyQualScoreTrain(frame.mvKeyQualScoreTrain),
     mvChi2(frame.mvChi2), mvChi2Dof(frame.mvChi2Dof),
     mvpMapPointsComp(frame.mvpMapPointsComp)
{
    for(int i=0;i<FRAME_GRID_COLS;i++)
        for(int j=0; j<FRAME_GRID_ROWS; j++)
            mGrid[i][j]=frame.mGrid[i][j];

    if(!frame.mTcw.empty())
        SetPose(frame.mTcw);
    if(!frame.mTwc_gt.empty())
        SetGroundTruthPose(frame.mTwc_gt);
    
    if(frame.mbPoseUncertaintyAvailable) {
        mSigmaTwc_gt = frame.mSigmaTwc_gt;
        mbPoseUncertaintyAvailable = true;
    }
    
    mvKeyQualScoreTrain = frame.mvKeyQualScoreTrain;
    mvChi2 = frame.mvChi2;
    mvChi2Dof = frame.mvChi2Dof;
    mvpMapPointsComp = frame.mvpMapPointsComp;
}


Frame::Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double 
&timeStamp, ORBextractor* extractorLeft, ORBextractor* extractorRight, 
ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float 
&thDepth, const bool &gtDepthAvailable, const cv::Mat &imDepth,
   const bool &bLogOutliers)
    
:mpORBvocabulary(voc),mpORBextractorLeft(extractorLeft),mpORBextractorRight(
extractorRight), mTimeStamp(timeStamp), 
mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
     mpReferenceKF(static_cast<KeyFrame*>(NULL))
{
    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
    // If predicted costmap for the image is available provide it to the
    // ORB extractor
    if (!imDepth.empty() && !gtDepthAvailable) { 
      thread threadLeft(&Frame::ExtractORBWeighted,this,0,imLeft, imDepth);
      thread threadRight(&Frame::ExtractORBWeighted,this,1,imRight, imDepth);
      threadLeft.join();
      threadRight.join();
    } else {
      thread threadLeft(&Frame::ExtractORB,this,0,imLeft);
      thread threadRight(&Frame::ExtractORB,this,1,imRight);
      threadLeft.join();
      threadRight.join();
    }
   

    N = mvKeys.size();
    // Initialize keypoint quality scores    
    if (!imDepth.empty() && !gtDepthAvailable) {
      for (int i = 0; i < N; i++) {
        int px = static_cast<int>(std::round(mvKeys[i].pt.x));
        int py = static_cast<int>(std::round(mvKeys[i].pt.y));
        float cost = static_cast<float>(imDepth.at<uint8_t>(py, px));
        float qual_score = 1.0 / (1.0 + cost/256);
        float qual_score_norm = 2 * qual_score - 1;
        mvKeyQualScore.push_back(qual_score_norm);
      }
    } else {
      for (int i = 0; i < N; i++) {
        mvKeyQualScore.push_back(1.0);
      }
    }

    if(mvKeys.empty())
        return;

    UndistortKeyPoints();
    
    // Set the ground truth keypoint depth values if the ground truth depthmap
    // is available
    if (!imDepth.empty() && gtDepthAvailable) {
      // Mask out the sky from the ground truth depth image
      const float kMaxDepth = 600.0; // meters
      cv::Mat depth_img_mask;
     
      cv::threshold(imDepth, 
                depth_img_mask, 
                kMaxDepth,
                1.0, 
                cv::THRESH_BINARY_INV);
     
      
      for (int i = 0; i < N;i++) {
        int px = static_cast<int>(std::round(mvKeysUn[i].pt.x));
        int py = static_cast<int>(std::round(mvKeysUn[i].pt.y));
//         mvKeysGTDepth.push_back(imDepth.at<float>(py, px));
        
        // TODO: Implement a more robust estimate of the depth of the patch.
//         float size = mvKeysUn[i].size;
        float size = 5;
//         float size = std::max(static_cast<float>(5.0), 
//                               static_cast<float>(0.2 * mvKeysUn[i].size));
        
        cv::Rect patch_def(px - std::floor(size/2.0), 
                           py - std::floor(size/2.0), 
                           size, size);
        cv::Mat patch = imDepth(patch_def);
        cv::Mat mask_patch = depth_img_mask(patch_def);
        double patch_depth;
        
        // Take median depth as depth of the patch
//         patch_depth = GetMaskedMedian(patch, mask_patch);
        
        // Take min depth as depth of the patch
        cv::minMaxLoc(patch, &patch_depth);
       
        
        mvKeysGTDepth.push_back(static_cast<float>(patch_depth));
      }
    }

    ComputeStereoMatches();

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));    
    mvbOutlier = vector<bool>(N,false);
   
    if(bLogOutliers) {
      mvpMapPointsComp = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
      mvChi2Dof = vector<int>(N, 0);
      mvChi2 = vector<float>(N, 0);
      mvKeyQualScoreTrain = vector<float>(N, 1.0f);
    }


    // This is done only for the first Frame (or after a change in the 
    // calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imLeft);

        
mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/(mnMaxX-mnMinX);
        
mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/(mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf/fx;

    AssignFeaturesToGrid();
}

Frame::Frame(const cv::Mat &imGray, const cv::Mat &imDepth, const double 
&timeStamp, ORBextractor* extractor,ORBVocabulary* voc, cv::Mat &K, cv::Mat 
&distCoef, const float &bf, const float &thDepth,const bool &bLogOutliers)
    
:mpORBvocabulary(voc),mpORBextractorLeft(extractor),mpORBextractorRight(
static_cast<ORBextractor*>(NULL)),
     mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), 
mThDepth(thDepth)
{
    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();    
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
    ExtractORB(0,imGray);

    N = mvKeys.size();
    // Initialize keypoint quality scores
    for (int i = 0; i < N; i++) {
      mvKeyQualScore.push_back(1.0);
    }

    if(mvKeys.empty())
        return;

    UndistortKeyPoints();

    ComputeStereoFromRGBD(imDepth);

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
    mvbOutlier = vector<bool>(N,false);
   
    if(bLogOutliers) {
      mvpMapPointsComp = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
      mvChi2Dof = vector<int>(N, 0);
      mvChi2 = vector<float>(N, 0);
      mvKeyQualScoreTrain = vector<float>(N, 1.0f);
    }

    // This is done only for the first Frame (or after a change in the 
    // calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imGray);

        
mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(
mnMaxX-mnMinX);
        
mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(
mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf/fx;

    AssignFeaturesToGrid();
}


Frame::Frame(const cv::Mat &imGray, const double &timeStamp, ORBextractor* 
extractor,ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, 
const float &thDepth, const bool &gtDepthAvailable, const cv::Mat &imDepth,
  const bool &bLogOutliers)
    
:mpORBvocabulary(voc),mpORBextractorLeft(extractor),mpORBextractorRight(
static_cast<ORBextractor*>(NULL)),
     mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), 
mThDepth(thDepth)
{
    // Frame ID
    mnId=nNextId++;

    // Scale Level Info
    mnScaleLevels = mpORBextractorLeft->GetLevels();
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
    mfLogScaleFactor = log(mfScaleFactor);
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
    if (!imDepth.empty() && !gtDepthAvailable) {
      ExtractORBWeighted(0, imGray, imDepth);
    } else {
      ExtractORB(0,imGray);
    }
    

    N = mvKeys.size();
    // Initialize keypoint quality scores
   
    if (!imDepth.empty() && !gtDepthAvailable) {
      for (int i = 0; i < N; i++) {
        int px = static_cast<int>(std::round(mvKeys[i].pt.x));
        int py = static_cast<int>(std::round(mvKeys[i].pt.y));
//         cout << px << ", " << py << ": ";
        float cost = static_cast<float>(imDepth.at<uint8_t>(py, px));
        float qual_score = 1.0 / (1.0 + cost/256);
        mvKeyQualScore.push_back(2 * qual_score - 1);
//         cout << cost << ": " << mvKeyQualScore[i]<< endl;
      }
    } else {
      for (int i = 0; i < N; i++) {
        mvKeyQualScore.push_back(1.0);
      }
    }
   

    if(mvKeys.empty())
        return;

    UndistortKeyPoints();
    
    // Set the ground truth keypoint depth values if the ground truth depthmap
    // is available
    if (!imDepth.empty() && gtDepthAvailable) {
      for (int i = 0; i < N;i++) {
        int px = static_cast<int>(std::round(mvKeysUn[i].pt.x));
        int py = static_cast<int>(std::round(mvKeysUn[i].pt.y));
        mvKeysGTDepth.push_back(imDepth.at<float>(py, px));
      }
    }

    // Set no stereo information
    mvuRight = vector<float>(N,-1);
    mvDepth = vector<float>(N,-1);

    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
    mvbOutlier = vector<bool>(N,false);
   
    if(bLogOutliers) {
      mvpMapPointsComp = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
      mvChi2Dof = vector<int>(N, 0);
      mvChi2 = vector<float>(N, 0);
      mvKeyQualScoreTrain = vector<float>(N, 1.0f);
    }
      
    // This is done only for the first Frame (or after a change in the 
    // calibration)
    if(mbInitialComputations)
    {
        ComputeImageBounds(imGray);

        
mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(
mnMaxX-mnMinX);
        
mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(
mnMaxY-mnMinY);

        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

        mbInitialComputations=false;
    }

    mb = mbf/fx;

    AssignFeaturesToGrid();
}

void Frame::AssignFeaturesToGrid()
{
    int nReserve = 0.5f*N/(FRAME_GRID_COLS*FRAME_GRID_ROWS);
    for(unsigned int i=0; i<FRAME_GRID_COLS;i++)
        for (unsigned int j=0; j<FRAME_GRID_ROWS;j++)
            mGrid[i][j].reserve(nReserve);

    for(int i=0;i<N;i++)
    {
        const cv::KeyPoint &kp = mvKeysUn[i];

        int nGridPosX, nGridPosY;
        if(PosInGrid(kp,nGridPosX,nGridPosY))
            mGrid[nGridPosX][nGridPosY].push_back(i);
    }
}

void Frame::ExtractORB(int flag, const cv::Mat &im)
{
    if(flag==0)
        (*mpORBextractorLeft)(im,cv::Mat(),mvKeys,mDescriptors);
    else
        (*mpORBextractorRight)(im,cv::Mat(),mvKeysRight,mDescriptorsRight);
}

void Frame::ExtractORBWeighted(int flag, 
                               const cv::Mat &im, 
                               const cv::Mat &costmap)
{
    if(flag==0)
        (*mpORBextractorLeft)(im,costmap,mvKeys,mDescriptors);
    else
        (*mpORBextractorRight)(im,costmap,mvKeysRight,mDescriptorsRight);
}

void Frame::SetPose(cv::Mat Tcw)
{
    mTcw = Tcw.clone();
    UpdatePoseMatrices();
}

void Frame::SetGroundTruthPose(cv::Mat Twc_gt) {
    mTwc_gt = Twc_gt.clone();
   
    cv::Mat Rwc_gt = mTwc_gt.rowRange(0,3).colRange(0,3);
    cv::Mat twc_gt = mTwc_gt.rowRange(0,3).col(3);
    cv::Mat Rcw_gt = Rwc_gt.t();
    cv::Mat tcw_gt = -Rcw_gt * twc_gt;
    
    mTcw_gt = cv::Mat::eye(4,4,mTwc_gt.type());
    Rcw_gt.copyTo(mTcw_gt.rowRange(0,3).colRange(0,3));
    tcw_gt.copyTo(mTcw_gt.rowRange(0,3).col(3));
}

void Frame::SetGroundTruthPose(cv::Mat Twc_gt, 
                               const Eigen::Matrix<double, 6, 6>& SigmaTwc_gt) {
    mTwc_gt = Twc_gt.clone();
    mSigmaTwc_gt = SigmaTwc_gt;
    mbPoseUncertaintyAvailable = true;

    cv::Mat Rwc_gt = mTwc_gt.rowRange(0,3).colRange(0,3);
    cv::Mat twc_gt = mTwc_gt.rowRange(0,3).col(3);
    cv::Mat Rcw_gt = Rwc_gt.t();
    cv::Mat tcw_gt = -Rcw_gt * twc_gt;
    
    mTcw_gt = cv::Mat::eye(4,4,mTwc_gt.type());
    Rcw_gt.copyTo(mTcw_gt.rowRange(0,3).colRange(0,3));
    tcw_gt.copyTo(mTcw_gt.rowRange(0,3).col(3));
}

void Frame::ApplyReferencePose() {
    mTcw = mTcw_gt.clone();
    UpdatePoseMatrices();
}

void Frame::BackupNewMapPoints() {
  for (size_t i = 0; i < mvChi2Dof.size(); i++) {
    if (mvChi2Dof[i] > 0 && !mvpMapPointsComp[i]) {
      mvpMapPointsComp[i] = mvpMapPoints[i];
    }
  }
}

void Frame::ComputeKeyPtQualScores() {
  const float prob_thresh_low = 0.5; // 0.5
  const int min_obs = 3;
  boost::math::chi_squared_distribution<float> dist_mono(2);
  boost::math::chi_squared_distribution<float> dist_stereo(3);
  
  float thresh_high_mono = quantile(dist_mono, 
                                    FLAGS_ivslam_keypt_qual_chi2_prob_thresh);
  float thresh_high_stereo = quantile(dist_stereo, 
                                      FLAGS_ivslam_keypt_qual_chi2_prob_thresh);
  float thresh_low_mono = quantile(dist_mono, prob_thresh_low);
  float thresh_low_stereo = quantile(dist_stereo, prob_thresh_low);
  
  for(size_t i = 0; i < mvChi2Dof.size(); i++) {
    if (mvChi2Dof[i] > 0) {
      if(!mvpMapPointsComp[i]) {
        LOG(FATAL) << "Map point not available";
      }
      
      float thresh_min, thresh_max;
      if (mvChi2Dof[i] == 2) {
        thresh_min = thresh_low_mono;
        thresh_max = thresh_high_mono;
      } else if(mvChi2Dof[i] == 3) {
        thresh_min = thresh_low_stereo;
        thresh_max = thresh_high_stereo;
      } else {
        LOG(FATAL) << "Unexpected Chi2DOF " << mvChi2Dof[i]; 
      }
      
      float chi2 = mvChi2[i];
      float scaled_err = (chi2 - thresh_min) / (thresh_max - thresh_min);
      scaled_err = (scaled_err > 1.0)? 1.0: scaled_err;
      scaled_err = (scaled_err < 0.0)? 0.0: scaled_err;
      
      float qual_score = 1.0 / (1.0 + static_cast<float>(scaled_err));
      float qual_score_norm = 2 * qual_score - 1;
      mvKeyQualScoreTrain[i] = qual_score_norm;
      mvpMapPointsComp[i]->SetQualityScore(qual_score_norm);
      
      // Prune out points that have a short track length and have been 
      // estimated to have good quality
      if(mvpMapPointsComp[i]->GetFound() < min_obs && 
         qual_score_norm > 0.5) {
        mvChi2Dof[i] = 0;
        continue;
      }
    }
  }
}

void Frame::UpdatePoseMatrices()
{ 
    mRcw = mTcw.rowRange(0,3).colRange(0,3);
    mRwc = mRcw.t();
    mtcw = mTcw.rowRange(0,3).col(3);
    mOw = -mRcw.t()*mtcw;
}

bool Frame::isInFrustum(MapPoint *pMP, float viewingCosLimit)
{
    pMP->mbTrackInView = false;

    // 3D in absolute coordinates
    cv::Mat P = pMP->GetWorldPos(); 

    // 3D in camera coordinates
    const cv::Mat Pc = mRcw*P+mtcw;
    const float &PcX = Pc.at<float>(0);
    const float &PcY= Pc.at<float>(1);
    const float &PcZ = Pc.at<float>(2);

    // Check positive depth
    if(PcZ<0.0f)
        return false;

    // Project in image and check it is not outside
    const float invz = 1.0f/PcZ;
    const float u=fx*PcX*invz+cx;
    const float v=fy*PcY*invz+cy;

    if(u<mnMinX || u>mnMaxX)
        return false;
    if(v<mnMinY || v>mnMaxY)
        return false;

    // Check distance is in the scale invariance region of the MapPoint
    const float maxDistance = pMP->GetMaxDistanceInvariance();
    const float minDistance = pMP->GetMinDistanceInvariance();
    const cv::Mat PO = P-mOw;
    const float dist = cv::norm(PO);

    if(dist<minDistance || dist>maxDistance)
        return false;

   // Check viewing angle
    cv::Mat Pn = pMP->GetNormal();

    const float viewCos = PO.dot(Pn)/dist;

    if(viewCos<viewingCosLimit)
        return false;

    // Predict scale in the image
    const int nPredictedLevel = pMP->PredictScale(dist,this);

    // Data used by the tracking
    pMP->mbTrackInView = true;
    pMP->mTrackProjX = u;
    pMP->mTrackProjXR = u - mbf*invz;
    pMP->mTrackProjY = v;
    pMP->mnTrackScaleLevel= nPredictedLevel;
    pMP->mTrackViewCos = viewCos;

    return true;
}

vector<size_t> Frame::GetFeaturesInArea(const float &x, const float  &y, const float  &r, const int minLevel, const int maxLevel) const
{
    vector<size_t> vIndices;
    vIndices.reserve(N);

    const int nMinCellX = max(0,(int)floor((x-mnMinX-r)*mfGridElementWidthInv));
    if(nMinCellX>=FRAME_GRID_COLS)
        return vIndices;

    const int nMaxCellX = min((int)FRAME_GRID_COLS-1,(int)ceil((x-mnMinX+r)*mfGridElementWidthInv));
    if(nMaxCellX<0)
        return vIndices;

    const int nMinCellY = max(0,(int)floor((y-mnMinY-r)*mfGridElementHeightInv));
    if(nMinCellY>=FRAME_GRID_ROWS)
        return vIndices;

    const int nMaxCellY = min((int)FRAME_GRID_ROWS-1,(int)ceil((y-mnMinY+r)*mfGridElementHeightInv));
    if(nMaxCellY<0)
        return vIndices;

    const bool bCheckLevels = (minLevel>0) || (maxLevel>=0);

    for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
    {
        for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
        {
            const vector<size_t> vCell = mGrid[ix][iy];
            if(vCell.empty())
                continue;

            for(size_t j=0, jend=vCell.size(); j<jend; j++)
            {
                const cv::KeyPoint &kpUn = mvKeysUn[vCell[j]];
                if(bCheckLevels)
                {
                    if(kpUn.octave<minLevel)
                        continue;
                    if(maxLevel>=0)
                        if(kpUn.octave>maxLevel)
                            continue;
                }

                const float distx = kpUn.pt.x-x;
                const float disty = kpUn.pt.y-y;

                if(fabs(distx)<r && fabs(disty)<r)
                    vIndices.push_back(vCell[j]);
            }
        }
    }

    return vIndices;
}

bool Frame::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY)
{
    posX = round((kp.pt.x-mnMinX)*mfGridElementWidthInv);
    posY = round((kp.pt.y-mnMinY)*mfGridElementHeightInv);

    //Keypoint's coordinates are undistorted, which could cause to go out of the image
    if(posX<0 || posX>=FRAME_GRID_COLS || posY<0 || posY>=FRAME_GRID_ROWS)
        return false;

    return true;
}


void Frame::ComputeBoW()
{
  // BOW computation can optionally be turned off if we do not want to 
  // use loop closure (if no ORB vocabulary is available)
  if (mpORBvocabulary) {
    if(mBowVec.empty())
    {
        vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
        mpORBvocabulary->transform(vCurrentDesc,mBowVec,mFeatVec,4);
    }
  }
}

void Frame::UndistortKeyPoints()
{
    if(mDistCoef.at<float>(0)==0.0)
    {
        mvKeysUn=mvKeys;
        return;
    }

    // Fill matrix with points
    cv::Mat mat(N,2,CV_32F);
    for(int i=0; i<N; i++)
    {
        mat.at<float>(i,0)=mvKeys[i].pt.x;
        mat.at<float>(i,1)=mvKeys[i].pt.y;
    }

    // Undistort points
    mat=mat.reshape(2);
    cv::undistortPoints(mat,mat,mK,mDistCoef,cv::Mat(),mK);
    mat=mat.reshape(1);

    // Fill undistorted keypoint vector
    mvKeysUn.resize(N);
    for(int i=0; i<N; i++)
    {
        cv::KeyPoint kp = mvKeys[i];
        kp.pt.x=mat.at<float>(i,0);
        kp.pt.y=mat.at<float>(i,1);
        mvKeysUn[i]=kp;
    }
}

void Frame::ComputeImageBounds(const cv::Mat &imLeft)
{
    if(mDistCoef.at<float>(0)!=0.0)
    {
        cv::Mat mat(4,2,CV_32F);
        mat.at<float>(0,0)=0.0; mat.at<float>(0,1)=0.0;
        mat.at<float>(1,0)=imLeft.cols; mat.at<float>(1,1)=0.0;
        mat.at<float>(2,0)=0.0; mat.at<float>(2,1)=imLeft.rows;
        mat.at<float>(3,0)=imLeft.cols; mat.at<float>(3,1)=imLeft.rows;

        // Undistort corners
        mat=mat.reshape(2);
        cv::undistortPoints(mat,mat,mK,mDistCoef,cv::Mat(),mK);
        mat=mat.reshape(1);

        mnMinX = min(mat.at<float>(0,0),mat.at<float>(2,0));
        mnMaxX = max(mat.at<float>(1,0),mat.at<float>(3,0));
        mnMinY = min(mat.at<float>(0,1),mat.at<float>(1,1));
        mnMaxY = max(mat.at<float>(2,1),mat.at<float>(3,1));

    }
    else
    {
        mnMinX = 0.0f;
        mnMaxX = imLeft.cols;
        mnMinY = 0.0f;
        mnMaxY = imLeft.rows;
    }
}

void Frame::ComputeStereoMatches()
{
    mvuRight = vector<float>(N,-1.0f);
    mvDepth = vector<float>(N,-1.0f);

    const int thOrbDist = (ORBmatcher::TH_HIGH+ORBmatcher::TH_LOW)/2;

    const int nRows = mpORBextractorLeft->mvImagePyramid[0].rows;

    //Assign keypoints to row table
    vector<vector<size_t> > vRowIndices(nRows,vector<size_t>());

    for(int i=0; i<nRows; i++)
        vRowIndices[i].reserve(200);

    const int Nr = mvKeysRight.size();

    for(int iR=0; iR<Nr; iR++)
    {
        const cv::KeyPoint &kp = mvKeysRight[iR];
        const float &kpY = kp.pt.y;
        const float r = 2.0f*mvScaleFactors[mvKeysRight[iR].octave];
        const int maxr = ceil(kpY+r);
        const int minr = floor(kpY-r);

        for(int yi=minr;yi<=maxr;yi++)
            vRowIndices[yi].push_back(iR);
    }

    // Set limits for search
    const float minZ = mb;
    const float minD = 0;
    const float maxD = mbf/minZ;

    // For each left keypoint search a match in the right image
    vector<pair<int, int> > vDistIdx;
    vDistIdx.reserve(N);

    for(int iL=0; iL<N; iL++)
    {
        const cv::KeyPoint &kpL = mvKeys[iL];
        const int &levelL = kpL.octave;
        const float &vL = kpL.pt.y;
        const float &uL = kpL.pt.x;

        const vector<size_t> &vCandidates = vRowIndices[vL];

        if(vCandidates.empty())
            continue;

        const float minU = uL-maxD;
        const float maxU = uL-minD;

        if(maxU<0)
            continue;

        int bestDist = ORBmatcher::TH_HIGH;
        size_t bestIdxR = 0;

        const cv::Mat &dL = mDescriptors.row(iL);

        // Compare descriptor to right keypoints
        for(size_t iC=0; iC<vCandidates.size(); iC++)
        {
            const size_t iR = vCandidates[iC];
            const cv::KeyPoint &kpR = mvKeysRight[iR];

            if(kpR.octave<levelL-1 || kpR.octave>levelL+1)
                continue;

            const float &uR = kpR.pt.x;

            if(uR>=minU && uR<=maxU)
            {
                const cv::Mat &dR = mDescriptorsRight.row(iR);
                const int dist = ORBmatcher::DescriptorDistance(dL,dR);

                if(dist<bestDist)
                {
                    bestDist = dist;
                    bestIdxR = iR;
                }
            }
        }

        // Subpixel match by correlation
        if(bestDist<thOrbDist)
        {
            // coordinates in image pyramid at keypoint scale
            const float uR0 = mvKeysRight[bestIdxR].pt.x;
            const float scaleFactor = mvInvScaleFactors[kpL.octave];
            const float scaleduL = round(kpL.pt.x*scaleFactor);
            const float scaledvL = round(kpL.pt.y*scaleFactor);
            const float scaleduR0 = round(uR0*scaleFactor);

            // sliding window search
            const int w = 5;
            cv::Mat IL = mpORBextractorLeft->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduL-w,scaleduL+w+1);
            IL.convertTo(IL,CV_32F);
            IL = IL - IL.at<float>(w,w) *cv::Mat::ones(IL.rows,IL.cols,CV_32F);

            int bestDist = INT_MAX;
            int bestincR = 0;
            const int L = 5;
            vector<float> vDists;
            vDists.resize(2*L+1);

            const float iniu = scaleduR0+L-w;
            const float endu = scaleduR0+L+w+1;
            if(iniu<0 || endu >= mpORBextractorRight->mvImagePyramid[kpL.octave].cols)
                continue;

            for(int incR=-L; incR<=+L; incR++)
            {
                cv::Mat IR = mpORBextractorRight->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduR0+incR-w,scaleduR0+incR+w+1);
                IR.convertTo(IR,CV_32F);
                IR = IR - IR.at<float>(w,w) *cv::Mat::ones(IR.rows,IR.cols,CV_32F);

                float dist = cv::norm(IL,IR,cv::NORM_L1);
                if(dist<bestDist)
                {
                    bestDist =  dist;
                    bestincR = incR;
                }

                vDists[L+incR] = dist;
            }

            if(bestincR==-L || bestincR==L)
                continue;

            // Sub-pixel match (Parabola fitting)
            const float dist1 = vDists[L+bestincR-1];
            const float dist2 = vDists[L+bestincR];
            const float dist3 = vDists[L+bestincR+1];

            const float deltaR = (dist1-dist3)/(2.0f*(dist1+dist3-2.0f*dist2));

            if(deltaR<-1 || deltaR>1)
                continue;

            // Re-scaled coordinate
            float bestuR = mvScaleFactors[kpL.octave]*((float)scaleduR0+(float)bestincR+deltaR);

            float disparity = (uL-bestuR);

            if(disparity>=minD && disparity<maxD)
            {
                if(disparity<=0)
                {
                    disparity=0.01;
                    bestuR = uL-0.01;
                }
                mvDepth[iL]=mbf/disparity;
                mvuRight[iL] = bestuR;
                vDistIdx.push_back(pair<int,int>(bestDist,iL));
            }
        }
    }

    sort(vDistIdx.begin(),vDistIdx.end());
    const float median = vDistIdx[vDistIdx.size()/2].first;
    const float thDist = 1.5f*1.4f*median;

    for(int i=vDistIdx.size()-1;i>=0;i--)
    {
        if(vDistIdx[i].first<thDist)
            break;
        else
        {
            mvuRight[vDistIdx[i].second]=-1;
            mvDepth[vDistIdx[i].second]=-1;
        }
    }
}


void Frame::ComputeStereoFromRGBD(const cv::Mat &imDepth)
{
    mvuRight = vector<float>(N,-1);
    mvDepth = vector<float>(N,-1);

    for(int i=0; i<N; i++)
    {
        const cv::KeyPoint &kp = mvKeys[i];
        const cv::KeyPoint &kpU = mvKeysUn[i];

        const float &v = kp.pt.y;
        const float &u = kp.pt.x;

        const float d = imDepth.at<float>(v,u);

        if(d>0)
        {
            mvDepth[i] = d;
            mvuRight[i] = kpU.pt.x-mbf/d;
        }
    }
}

cv::Mat Frame::UnprojectStereo(const int &i)
{
    const float z = mvDepth[i];
    if(z>0)
    {
        const float u = mvKeysUn[i].pt.x;
        const float v = mvKeysUn[i].pt.y;
        const float x = (u-cx)*z*invfx;
        const float y = (v-cy)*z*invfy;
        cv::Mat x3Dc = (cv::Mat_<float>(3,1) << x, y, z);
        return mRwc*x3Dc+mOw;
    }
    else
        return cv::Mat();
}

bool Frame::GetCorrespondingKeyPt(MapPoint* map_point, 
                                 cv::KeyPoint* keypoint) {
    std::vector<MapPoint*>::iterator it;
    it = find(mvpMapPoints.begin(), mvpMapPoints.end(), map_point);
    if (it != mvpMapPoints.end()) {
        *keypoint = mvKeysUn[it - mvpMapPoints.begin()];
        return true;
    } else {
        return false;
    }
}

bool Frame::GetCorrespondingKeyPt(MapPoint* map_point, 
                                 cv::KeyPoint* keypoint,
                                 int* idx) {
    std::vector<MapPoint*>::iterator it;
    it = find(mvpMapPoints.begin(), mvpMapPoints.end(), map_point);
    if (it != mvpMapPoints.end()) {
        *keypoint = mvKeysUn[it - mvpMapPoints.begin()];
        *idx = it - mvpMapPoints.begin();
        return true;
    } else {
        return false;
    }
}

double Frame::GetMaskedMedian(const cv::Mat& input_img, const cv::Mat& mask) {
  vector<float> points;
  for (size_t i = 0; i < input_img.rows; i++) {
    for (size_t j = 0; j < input_img.cols; j++) {
      if (mask.at<float>(i, j)) {
        points.push_back(input_img.at<float>(i, j));
      }
    }
  }
  
  sort(points.begin(), points.end()); 
  int points_num = points.size();
  
  if (points.empty()) {
    LOG(WARNING) << "All points in the patch are masked!"; 
    return cv::mean(input_img)[0];
  }
  
  if (points_num % 2 != 0) {
    return static_cast<double>(points[std::floor(points_num / 2)]); 
  } else {
    return static_cast<double>((points[(points_num / 2) - 1] + 
        points[points_num / 2]) / 2.0); 
  }
}

} //namespace ORB_SLAM
