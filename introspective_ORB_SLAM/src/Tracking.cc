/**
 * This file is part of ORB-SLAM2.
 *
 * Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University
 * of Zaragoza) For more information see <https://github.com/raulmur/ORB_SLAM2>
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

#include "Tracking.h"

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#if (CV_VERSION_MAJOR >= 4)
#include <opencv2/imgcodecs/legacy/constants_c.h>
#include <opencv2/imgproc/imgproc_c.h>
#endif

#include <glog/logging.h>

#include <algorithm>
#include <iostream>
#include <mutex>

#include "Converter.h"
#include "FrameDrawer.h"
#include "Initializer.h"
#include "Map.h"
#include "ORBmatcher.h"
#include "Optimizer.h"
#include "PnPsolver.h"

DEFINE_int32(tracking_ba_rate,
             1,
             "Rate of running local bundle adjustemnt in single threaded mode."
             " BA will run each time a new key frame is generated if at least "
             "tracking_ba_rate number of frames have been receive since last "
             "BA execution.");

DEFINE_double(ivslam_ref_pose_ang_var_inv,
              2.0e+3,
              "Inverse of the variance of the reference camera "
              "orientation with an angle-axis representation. This is for "
              "evaluating the accuracy of SLAM's tracking in the unsupervised "
              "learning mode and in order to determine whether a frame "
              "should be used for training.");
DEFINE_double(ivslam_ref_pose_trans_var_inv,
              2.0e+2,
              "Inverse of the variance of the reference camera "
              "position. This is for "
              "evaluating the accuracy of SLAM's tracking in the unsupervised "
              "learning mode and in order to determine whether a frame "
              "should be used for training.");

using namespace std;
using namespace feature_evaluation;

namespace ORB_SLAM2 {

Tracking::Tracking(System* pSys,
                   ORBVocabulary* pVoc,
                   FrameDrawer* pFrameDrawer,
                   MapDrawer* pMapDrawer,
                   Map* pMap,
                   KeyFrameDatabase* pKFDB,
                   const string& strSettingPath,
                   const int sensor,
                   const bool bSingleThreaded,
                   const bool bSilent,
                   const bool bGuidedBA)
    : mState(NO_IMAGES_YET),
      mSensor(sensor),
      mbSingleThreaded(bSingleThreaded),
      mbOnlyTracking(false),
      mbVO(false),
      mpORBVocabulary(pVoc),
      mpKeyFrameDB(pKFDB),
      mpInitializer(static_cast<Initializer*>(NULL)),
      mpSystem(pSys),
      mpViewer(NULL),
      mpFrameDrawer(pFrameDrawer),
      mpMapDrawer(pMapDrawer),
      mpMap(pMap),
      mnLastRelocFrameId(0),
      mbSilent(bSilent),
      mbGuidedBA(bGuidedBA) {
  // Load camera parameters from settings file

  cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
  float fx = fSettings["Camera.fx"];
  float fy = fSettings["Camera.fy"];
  float cx = fSettings["Camera.cx"];
  float cy = fSettings["Camera.cy"];

  cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
  K.at<float>(0, 0) = fx;
  K.at<float>(1, 1) = fy;
  K.at<float>(0, 2) = cx;
  K.at<float>(1, 2) = cy;
  K.copyTo(mK);

  cv::Mat DistCoef(4, 1, CV_32F);
  DistCoef.at<float>(0) = fSettings["Camera.k1"];
  DistCoef.at<float>(1) = fSettings["Camera.k2"];
  DistCoef.at<float>(2) = fSettings["Camera.p1"];
  DistCoef.at<float>(3) = fSettings["Camera.p2"];
  const float k3 = fSettings["Camera.k3"];
  if (k3 != 0) {
    DistCoef.resize(5);
    DistCoef.at<float>(4) = k3;
  }
  DistCoef.copyTo(mDistCoef);

  mbf = fSettings["Camera.bf"];

  float fps = fSettings["Camera.fps"];
  if (fps == 0) fps = 30;

  // Max/Min Frames to insert keyframes and to check relocalisation
  mMinFrames = 0;
  mMaxFrames = fps;

  if (!bSilent) {
    cout << endl << "Camera Parameters: " << endl;
    cout << "- fx: " << fx << endl;
    cout << "- fy: " << fy << endl;
    cout << "- cx: " << cx << endl;
    cout << "- cy: " << cy << endl;
    cout << "- k1: " << DistCoef.at<float>(0) << endl;
    cout << "- k2: " << DistCoef.at<float>(1) << endl;
    if (DistCoef.rows == 5) cout << "- k3: " << DistCoef.at<float>(4) << endl;
    cout << "- p1: " << DistCoef.at<float>(2) << endl;
    cout << "- p2: " << DistCoef.at<float>(3) << endl;
    cout << "- fps: " << fps << endl;
  }

  int nRGB = fSettings["Camera.RGB"];
  mbRGB = nRGB;

  if (!bSilent) {
    if (mbRGB)
      cout << "- color order: RGB (ignored if grayscale)" << endl;
    else
      cout << "- color order: BGR (ignored if grayscale)" << endl;
  }

  // Load ORB parameters

  int nFeatures = fSettings["ORBextractor.nFeatures"];
  float fScaleFactor = fSettings["ORBextractor.scaleFactor"];
  int nLevels = fSettings["ORBextractor.nLevels"];
  int fIniThFAST = fSettings["ORBextractor.iniThFAST"];
  int fMinThFAST = fSettings["ORBextractor.minThFAST"];
  bool enableInrospectiveFeatExtraction = false;

  cv::FileNode introspective_ext =
      fSettings["ORBextractor.enableIntrospection"];
  if (!introspective_ext.empty()) {
    enableInrospectiveFeatExtraction =
        static_cast<bool>(int(fSettings["ORBextractor.enableIntrospection"]));
  }

  mpORBextractorLeft = new ORBextractor(nFeatures,
                                        fScaleFactor,
                                        nLevels,
                                        fIniThFAST,
                                        fMinThFAST,
                                        enableInrospectiveFeatExtraction);

  if (sensor == System::STEREO)
    mpORBextractorRight = new ORBextractor(
        nFeatures, fScaleFactor, nLevels, fIniThFAST, fMinThFAST);

  if (sensor == System::MONOCULAR)
    mpIniORBextractor = new ORBextractor(2 * nFeatures,
                                         fScaleFactor,
                                         nLevels,
                                         fIniThFAST,
                                         fMinThFAST,
                                         enableInrospectiveFeatExtraction);

  if (!bSilent) {
    cout << endl << "ORB Extractor Parameters: " << endl;
    cout << "- Number of Features: " << nFeatures << endl;
    cout << "- Scale Levels: " << nLevels << endl;
    cout << "- Scale Factor: " << fScaleFactor << endl;
    cout << "- Initial Fast Threshold: " << fIniThFAST << endl;
    cout << "- Minimum Fast Threshold: " << fMinThFAST << endl;
    cout << "- Intorspective Feature Extraction: "
         << enableInrospectiveFeatExtraction << endl;
  }

  if (sensor == System::STEREO || sensor == System::RGBD) {
    mThDepth = mbf * (float)fSettings["ThDepth"] / fx;

    if (!bSilent) {
      cout << endl
           << "Depth Threshold (Close/Far Points): " << mThDepth << endl;
    }
  }

  if (sensor == System::RGBD) {
    mDepthMapFactor = fSettings["DepthMapFactor"];
    if (fabs(mDepthMapFactor) < 1e-5)
      mDepthMapFactor = 1;
    else
      mDepthMapFactor = 1.0f / mDepthMapFactor;
  }

  // Load ORB Matcher params
  cv::FileNode nnratio = fSettings["ORBMatcher.NNRatioMultiplier"];
  cv::FileNode search_window = fSettings["ORBMatcher.SearchWindowMultiplier"];
  if (!nnratio.empty()) {
    mMatcherNNRatioMultiplier = fSettings["ORBMatcher.NNRatioMultiplier"];
    if (!bSilent) {
      cout << "ORBMatcher.NNRatioMultiplier: " << mMatcherNNRatioMultiplier
           << endl;
    }
  }

  if (!search_window.empty()) {
    mMatcherSearchWindowMultiplier =
        fSettings["ORBMatcher.SearchWindowMultiplier"];
    if (!bSilent) {
      cout << "ORBMatcher.SearchWindowMultiplier: "
           << mMatcherSearchWindowMultiplier << endl
           << endl;
    }
  }

  // IVSLAM training parameters
  cv::FileNode unsupervised_learning = fSettings["IVSLAM.unsupervisedLearning"];
  if (!unsupervised_learning.empty()) {
    mbUnsupervisedLearning =
        static_cast<bool>(int(fSettings["IVSLAM.unsupervisedLearning"]));
  } else {
    mbUnsupervisedLearning = false;
  }

  if (!bSilent) {
    cout << endl << "IV-SLAM Training Parameters: " << endl;
    cout << "- Unsupervised Learning: " << mbUnsupervisedLearning << endl;
  }

  mFeatureEvaluator = new FeatureEvaluator(kORB, kKITTI);
  if (sensor == System::STEREO) {
    mFeatureEvaluator->LoadRectificationMap(strSettingPath);
  }
}

void Tracking::SetLocalMapper(LocalMapping* pLocalMapper) {
  mpLocalMapper = pLocalMapper;
}

void Tracking::SetLoopClosing(LoopClosing* pLoopClosing) {
  mpLoopClosing = pLoopClosing;
}

void Tracking::SetViewer(Viewer* pViewer) { mpViewer = pViewer; }

cv::Mat Tracking::GrabImageStereo(const cv::Mat& imRectLeft,
                                  const cv::Mat& imRectRight,
                                  const double& timestamp) {
  mImGray = imRectLeft;
  cv::Mat imGrayRight = imRectRight;

  if (mImGray.channels() == 3) {
    if (mbRGB) {
      cvtColor(mImGray, mImGray, CV_RGB2GRAY);
      cvtColor(imGrayRight, imGrayRight, CV_RGB2GRAY);
    } else {
      cvtColor(mImGray, mImGray, CV_BGR2GRAY);
      cvtColor(imGrayRight, imGrayRight, CV_BGR2GRAY);
    }
  } else if (mImGray.channels() == 4) {
    if (mbRGB) {
      cvtColor(mImGray, mImGray, CV_RGBA2GRAY);
      cvtColor(imGrayRight, imGrayRight, CV_RGBA2GRAY);
    } else {
      cvtColor(mImGray, mImGray, CV_BGRA2GRAY);
      cvtColor(imGrayRight, imGrayRight, CV_BGRA2GRAY);
    }
  }

  mCurrentFrame = Frame(mImGray,
                        imGrayRight,
                        timestamp,
                        mpORBextractorLeft,
                        mpORBextractorRight,
                        mpORBVocabulary,
                        mK,
                        mDistCoef,
                        mbf,
                        mThDepth);

  Track();

  return mCurrentFrame.mTcw.clone();
}

cv::Mat Tracking::GrabImageStereo(
    const cv::Mat& imRectLeft,
    const cv::Mat& imRectRight,
    const double& timestamp,
    const cv::Mat& cam_pose_gt,
    const Eigen::Matrix<double, 6, 6>& cam_pose_gt_cov,
    const bool& pose_cov_available,
    const std::string& img_name,
    const bool& gtDepthAvailable,
    const cv::Mat& depthmap) {
  mImGray = imRectLeft;
  cv::Mat imGrayRight = imRectRight;

  if (mImGray.channels() == 3) {
    if (mbRGB) {
      cvtColor(mImGray, mImGray, CV_RGB2GRAY);
      cvtColor(imGrayRight, imGrayRight, CV_RGB2GRAY);
    } else {
      cvtColor(mImGray, mImGray, CV_BGR2GRAY);
      cvtColor(imGrayRight, imGrayRight, CV_BGR2GRAY);
    }
  } else if (mImGray.channels() == 4) {
    if (mbRGB) {
      cvtColor(mImGray, mImGray, CV_RGBA2GRAY);
      cvtColor(imGrayRight, imGrayRight, CV_RGBA2GRAY);
    } else {
      cvtColor(mImGray, mImGray, CV_BGRA2GRAY);
      cvtColor(imGrayRight, imGrayRight, CV_BGRA2GRAY);
    }
  }

  mCurrentFrame = Frame(mImGray,
                        imGrayRight,
                        timestamp,
                        mpORBextractorLeft,
                        mpORBextractorRight,
                        mpORBVocabulary,
                        mK,
                        mDistCoef,
                        mbf,
                        mThDepth,
                        gtDepthAvailable,
                        depthmap,
                        mbUnsupervisedLearning);

  if (mbTrainingMode) {
    if (!cam_pose_gt.empty()) {
      // Load the ground truth camera pose
      if (pose_cov_available) {
        mCurrentFrame.SetGroundTruthPose(cam_pose_gt, cam_pose_gt_cov);
      } else {
        mCurrentFrame.SetGroundTruthPose(cam_pose_gt);
      }
    }

    mCurrentFrame.mstrLeftImgName = img_name;
  }
  Track();

  return mCurrentFrame.mTcw.clone();
}

cv::Mat Tracking::GrabImageRGBD(const cv::Mat& imRGB,
                                const cv::Mat& imD,
                                const double& timestamp) {
  if (mbTrainingMode || mbIntrospectionOn) {
    LOG(FATAL) << "Introspective perception is not implemented for RGBD mode";
  }
  mImGray = imRGB;
  cv::Mat imDepth = imD;

  if (mImGray.channels() == 3) {
    if (mbRGB)
      cvtColor(mImGray, mImGray, CV_RGB2GRAY);
    else
      cvtColor(mImGray, mImGray, CV_BGR2GRAY);
  } else if (mImGray.channels() == 4) {
    if (mbRGB)
      cvtColor(mImGray, mImGray, CV_RGBA2GRAY);
    else
      cvtColor(mImGray, mImGray, CV_BGRA2GRAY);
  }

  if ((fabs(mDepthMapFactor - 1.0f) > 1e-5) || imDepth.type() != CV_32F)
    imDepth.convertTo(imDepth, CV_32F, mDepthMapFactor);

  mCurrentFrame = Frame(mImGray,
                        imDepth,
                        timestamp,
                        mpORBextractorLeft,
                        mpORBVocabulary,
                        mK,
                        mDistCoef,
                        mbf,
                        mThDepth);

  Track();

  return mCurrentFrame.mTcw.clone();
}

cv::Mat Tracking::GrabImageMonocular(const cv::Mat& im,
                                     const double& timestamp) {
  mImGray = im;

  if (mImGray.channels() == 3) {
    if (mbRGB)
      cvtColor(mImGray, mImGray, CV_RGB2GRAY);
    else
      cvtColor(mImGray, mImGray, CV_BGR2GRAY);
  } else if (mImGray.channels() == 4) {
    if (mbRGB)
      cvtColor(mImGray, mImGray, CV_RGBA2GRAY);
    else
      cvtColor(mImGray, mImGray, CV_BGRA2GRAY);
  }

  if (mState == NOT_INITIALIZED || mState == NO_IMAGES_YET)
    mCurrentFrame = Frame(mImGray,
                          timestamp,
                          mpIniORBextractor,
                          mpORBVocabulary,
                          mK,
                          mDistCoef,
                          mbf,
                          mThDepth);
  else
    mCurrentFrame = Frame(mImGray,
                          timestamp,
                          mpORBextractorLeft,
                          mpORBVocabulary,
                          mK,
                          mDistCoef,
                          mbf,
                          mThDepth);

  Track();

  return mCurrentFrame.mTcw.clone();
}

// The ground truth depth map is also an optional argument that could be
// passed to this function
cv::Mat Tracking::GrabImageMonocular(const cv::Mat& im,
                                     const double& timestamp,
                                     const cv::Mat& cam_pose_gt,
                                     const std::string& img_name,
                                     const bool& gtDepthAvailable,
                                     const cv::Mat& depthmap) {
  mImGray = im;
  mImDepth = depthmap;

  if (mImGray.channels() == 3) {
    if (mbRGB)
      cvtColor(mImGray, mImGray, CV_RGB2GRAY);
    else
      cvtColor(mImGray, mImGray, CV_BGR2GRAY);
  } else if (mImGray.channels() == 4) {
    if (mbRGB)
      cvtColor(mImGray, mImGray, CV_RGBA2GRAY);
    else
      cvtColor(mImGray, mImGray, CV_BGRA2GRAY);
  }

  if (mState == NOT_INITIALIZED || mState == NO_IMAGES_YET) {
    //         mCurrentFrame = Frame(mImGray,timestamp,mpIniORBextractor,
    //                               mpORBVocabulary,mK,mDistCoef,mbf,
    //                               mThDepth, gtDepthAvailable,
    //                               depthmap, mbUnsupervisedLearning);
    mCurrentFrame = Frame(mImGray,
                          timestamp,
                          mpIniORBextractor,
                          mpORBVocabulary,
                          mK,
                          mDistCoef,
                          mbf,
                          mThDepth);
  } else {
    mCurrentFrame = Frame(mImGray,
                          timestamp,
                          mpORBextractorLeft,
                          mpORBVocabulary,
                          mK,
                          mDistCoef,
                          mbf,
                          mThDepth,
                          gtDepthAvailable,
                          depthmap,
                          mbUnsupervisedLearning);
  }

  if (mbTrainingMode) {
    // Load the ground truth camera pose
    mCurrentFrame.SetGroundTruthPose(cam_pose_gt);
    mCurrentFrame.mstrLeftImgName = img_name;
  }
  Track();

  return mCurrentFrame.mTcw.clone();
}

void Tracking::Track() {
  mFramesReceivedSinceLastLocalBA++;

  if (mState == NO_IMAGES_YET) {
    mState = NOT_INITIALIZED;
  }

  mLastProcessedState = mState;

  // Get Map Mutex -> Map cannot be changed
  unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

  if (mState == NOT_INITIALIZED) {
    if (mSensor == System::STEREO || mSensor == System::RGBD) {
      StereoInitialization();
    } else {
      MonocularInitialization();
    }

    if (mpFrameDrawer) {
      mpFrameDrawer->Update(this);
    }

    if (mbUnsupervisedLearning) {
      CHECK_EQ(mbTrainingMode, true)
          << "Unsupervised Learning can only be set in Training mode!";
    }

    if (mState != OK) return;
  } else {
    // System is initialized. Track Frame.
    bool bOK;

    // Initial camera pose estimation using motion model or relocalization (if
    // tracking is lost)
    if (!mbOnlyTracking) {
      // Local Mapping is activated. This is the normal behaviour, unless
      // you explicitly activate the "only tracking" mode.

      if (mState == OK) {
        // Local Mapping might have changed some MapPoints tracked in last frame
        CheckReplacedInLastFrame();

        if (mVelocity.empty() || mCurrentFrame.mnId < mnLastRelocFrameId + 2) {
          if (mpORBVocabulary) {
            bOK = TrackReferenceKeyFrame();
          } else {
            // Run tracking with motion model with loosened matching
            // thresholds
            //                     bOK = TrackWithMotionModel(0.7, 10.0);
            bOK = TrackReferenceKeyFrame(0.9, 20.0, false);
          }
        } else {
          bOK = TrackWithMotionModel();
          // TODO: remove this testing +++
          //                     bOK = TrackReferenceKeyFrame();
          // ---

          if (!bOK) {
            if (mpORBVocabulary) {
              bOK = TrackReferenceKeyFrame();
            } else {
              // Run tracking with motion model with loosened
              // matching thresholds

              //                           bOK =
              //                           TrackWithMotionModel(0.7, 10.0);
              bOK = TrackReferenceKeyFrame(0.9, 20.0, false);
            }
          }
        }
      } else {
        std::cout << "Relocalization" << std::endl;
        bOK = Relocalization();
      }
    } else {
      // Localization Mode: Local Mapping is deactivated

      if (mState == LOST) {
        bOK = Relocalization();
      } else {
        if (!mbVO) {
          // In last frame we tracked enough MapPoints in the map

          if (!mVelocity.empty()) {
            bOK = TrackWithMotionModel();
          } else {
            bOK = TrackReferenceKeyFrame();
          }
        } else {
          // In last frame we tracked mainly "visual odometry" points.

          // We compute two camera poses, one from motion model and one doing
          // relocalization. If relocalization is sucessfull we choose that
          // solution, otherwise we retain the "visual odometry" solution.

          bool bOKMM = false;
          bool bOKReloc = false;
          vector<MapPoint*> vpMPsMM;
          vector<bool> vbOutMM;
          cv::Mat TcwMM;
          if (!mVelocity.empty()) {
            bOKMM = TrackWithMotionModel();
            vpMPsMM = mCurrentFrame.mvpMapPoints;
            vbOutMM = mCurrentFrame.mvbOutlier;
            TcwMM = mCurrentFrame.mTcw.clone();
          }
          bOKReloc = Relocalization();

          if (bOKMM && !bOKReloc) {
            mCurrentFrame.SetPose(TcwMM);
            mCurrentFrame.mvpMapPoints = vpMPsMM;
            mCurrentFrame.mvbOutlier = vbOutMM;

            if (mbVO) {
              for (int i = 0; i < mCurrentFrame.N; i++) {
                if (mCurrentFrame.mvpMapPoints[i] &&
                    !mCurrentFrame.mvbOutlier[i]) {
                  mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
                }
              }
            }
          } else if (bOKReloc) {
            mbVO = false;
          }

          bOK = bOKReloc || bOKMM;
        }
      }
    }

    mCurrentFrame.mpReferenceKF = mpReferenceKF;

    // If we have an initial estimation of the camera pose and matching. Track
    // the local map.
    if (!mbOnlyTracking) {
      if (bOK) {
        bOK = TrackLocalMap();
      }
    } else {
      // mbVO true means that there are few matches to MapPoints in the map. We
      // cannot retrieve a local map and therefore we do not perform
      // TrackLocalMap(). Once the system relocalizes the camera we will use the
      // local map again.
      if (bOK && !mbVO) {
        bOK = TrackLocalMap();
      }
    }

    if (mbUnsupervisedLearning) {
      mCurrentFrame.ComputeKeyPtQualScores();
    }

    if (bOK)
      mState = OK;
    else
      mState = LOST;

    // Update drawer
    if (mpFrameDrawer) {
      mpFrameDrawer->Update(this);
    }

    if (bOK) {
      if (mbTrainingMode) {
        mpMapDrawer->SetCurrentCameraPosewithGT(mCurrentFrame.mTcw,
                                                mCurrentFrame.mTwc_gt,
                                                mCurrentFrame.mstrLeftImgName);
      } else {
        mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw,
                                          mCurrentFrame.mstrLeftImgName);
      }
    }

    static int counter = 0;
    counter++;
    if (!mbIntrospectionOn && mbTrainingMode && counter < 2) {
      mFeatureEvaluator->UpdateCameraCalibration(mCurrentFrame);
    }
    if (!mbIntrospectionOn && mbTrainingMode && mState == OK && counter > 2) {
      // Evaluate the recent frames and calculate the quality heatmap
      mFeatureEvaluator->LoadImagePair(mImGrayPrev, mImGray);

      if (!mbUnsupervisedLearning) {
        mFeatureEvaluator->EvaluateFeatures(mLastFrame, mCurrentFrame);
        //           mFeatureEvaluator->GenerateImageQualityHeatmap();
        mFeatureEvaluator->GenerateImageQualityHeatmapGP();
      } else {
        bool tracking_reliable = EvaluateTrackingAccuracy();
        Reliability reliability = (tracking_reliable) ? Reliable : Unreliable;
        mFeatureEvaluator->SetFrameReliability(reliability);

        // Generate image quality heatmap from unsupervised quality
        // estimations
        mFeatureEvaluator->GenerateUnsupImageQualityHeatmapGP(mCurrentFrame);
        //           mFeatureEvaluator->GenerateUnsupImageQualityHeatmap(mCurrentFrame,
        //                                                   mvSaveVisualizationPath);

        // It is for running
        // feature evaluation on the full set of matched features including
        // the ones that have been pruned as outliers
        if (mbEnforceSupervisedFeatureEval) {
          mFeatureEvaluator->EvaluateFeatures(mLastFrame, mCurrentFrame);
        }
      }

      if (mbSaveVisualizationsToFile) {
        mFeatureEvaluator->SaveImagesToFile(mvSaveVisualizationPath,
                                            mCurrentFrame.mstrLeftImgName,
                                            mpMap->KeyFramesInMap() <= 15);
      }

      // Get the generated training data from feature evaluator and
      // use an instance of dataset_creator to write them to file
      if (mbCreateIntrospectionDataset) {
        // ***********************
        // The dataset of all images
        //             if(!mDatasetCreatorFull) {
        //               mDatasetCreatorFull =
        //                   new
        //                   DatasetCreator(mvOutputIntrospectionDatasetFullPath);
        //             }
        //             mDatasetCreatorFull->SaveBadRegionHeatmap(
        //                           mCurrentFrame.mstrLeftImgName,
        //                           mFeatureEvaluator->GetBadRegionHeatmap());

        // ***********************
        // The dataset of selected images for training the introspection
        // model
        if (mFeatureEvaluator->IsFrameGoodForTraining() ||
            (mbEnforceSupervisedFeatureEval &&
             mFeatureEvaluator->GetMatchedKeyPoints().size() > 0)) {
          if (!mDatasetCreator) {
            mDatasetCreator =
                new DatasetCreator(mvOutputIntrospectionDatasetPath);
          }

          // Organize when this should be called
          if (mbEnforceSupervisedFeatureEval) {
            mDatasetCreator->AppendKeypoints(
                mFeatureEvaluator->GetMatchedKeyPoints(),
                mFeatureEvaluator->GetErrValues());
          }

          mDatasetCreator->SaveBadRegionHeatmap(
              mCurrentFrame.mstrLeftImgName,
              mFeatureEvaluator->GetBadRegionHeatmap());

          if (mbUnsupervisedLearning) {
            mDatasetCreator->SaveBadRegionHeatmapMask(
                mCurrentFrame.mstrLeftImgName,
                mFeatureEvaluator->GetBadRegionHeatmapMask());
          }
        }
      }
    }

    if (!mbIntrospectionOn && mbTrainingMode) {
      mImGray.copyTo(mImGrayPrev);
    }

    // If tracking were good, check if we insert a keyframe
    if (bOK) {
      // Update motion model
      if (!mLastFrame.mTcw.empty()) {
        cv::Mat LastTwc = cv::Mat::eye(4, 4, CV_32F);
        mLastFrame.GetRotationInverse().copyTo(
            LastTwc.rowRange(0, 3).colRange(0, 3));
        mLastFrame.GetCameraCenter().copyTo(LastTwc.rowRange(0, 3).col(3));
        mVelocity = mCurrentFrame.mTcw * LastTwc;
      } else
        mVelocity = cv::Mat();

      // Clean VO matches
      for (int i = 0; i < mCurrentFrame.N; i++) {
        MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
        if (pMP)
          if (pMP->Observations() < 1) {
            mCurrentFrame.mvbOutlier[i] = false;
            mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
          }
      }

      // Delete temporal MapPoints
      for (list<MapPoint*>::iterator lit = mlpTemporalPoints.begin(),
                                     lend = mlpTemporalPoints.end();
           lit != lend;
           lit++) {
        MapPoint* pMP = *lit;
        delete pMP;
      }
      mlpTemporalPoints.clear();

      // Check if we need to insert a new keyframe
      if (NeedNewKeyFrame()) CreateNewKeyFrame();

      // TODO: Do not always run this visualization
      if (mbGuidedBA || false) {
        VisualizeKeyPtsQuality();
      }

      // We allow points with high innovation (considererd outliers by the Huber
      // Function) pass to the new keyframe, so that bundle adjustment will
      // finally decide if they are outliers or not. We don't want next frame to
      // estimate its position with those points so we discard them in the
      // frame.
      for (int i = 0; i < mCurrentFrame.N; i++) {
        if (mCurrentFrame.mvpMapPoints[i] && mCurrentFrame.mvbOutlier[i])
          mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
      }
    }

    // Reset if the camera get lost soon after initialization
    if (mState == LOST) {
      if (mbIntrospectionOn || true) {
        if (mpMap->KeyFramesInMap() <= 3) {
          LOG(WARNING) << "Track lost soon after initialisation, "
                          "reseting and not counting as new failure";
          mpSystem->Reset();
          return;
        } else {
          mFailureCount++;
          LOG(WARNING) << "Failure #" << mFailureCount << ". Resetting...";

          if (!mbSilent) {
            SaveTrackingResults(true);
          }
          mpSystem->Reset();
          return;
        }
      }

      if (mpMap->KeyFramesInMap() <= 5) {
        LOG(WARNING) << "Track lost soon after initialisation, "
                     << "reseting..." << endl;
        mpSystem->Reset();
        return;
      }
    }

    if (!mCurrentFrame.mpReferenceKF)
      mCurrentFrame.mpReferenceKF = mpReferenceKF;

    mLastFrame = Frame(mCurrentFrame);
  }

  // Store frame pose information to retrieve the complete camera trajectory
  // afterwards.
  if (!mCurrentFrame.mTcw.empty()) {
    cv::Mat Tcr =
        mCurrentFrame.mTcw * mCurrentFrame.mpReferenceKF->GetPoseInverse();
    mlRelativeFramePoses.push_back(Tcr);
    mlpReferences.push_back(mpReferenceKF);
    mlFrameTimes.push_back(mCurrentFrame.mTimeStamp);
    mlbLost.push_back(mState == LOST);

    mCurrentFramePose = CalculateInverseTransform(mCurrentFrame.mTcw);
  } else {
    // This can happen if tracking is lost
    mlRelativeFramePoses.push_back(mlRelativeFramePoses.back());
    mlpReferences.push_back(mlpReferences.back());
    mlFrameTimes.push_back(mlFrameTimes.back());
    mlbLost.push_back(mState == LOST);
  }
}

void Tracking::StereoInitialization() {
  if (mCurrentFrame.N > 500) {
    // Set Frame pose to the origin
    if (mbTrainingMode && mbGuidedBA) {
      mCurrentFrame.ApplyReferencePose();
    } else {
      mCurrentFrame.SetPose(cv::Mat::eye(4, 4, CV_32F));
    }

    // Create KeyFrame
    KeyFrame* pKFini = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);

    // Insert KeyFrame in the map
    mpMap->AddKeyFrame(pKFini);

    // Create MapPoints and asscoiate to KeyFrame
    for (int i = 0; i < mCurrentFrame.N; i++) {
      float z = mCurrentFrame.mvDepth[i];
      if (z > 0) {
        cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
        MapPoint* pNewMP = new MapPoint(x3D, pKFini, mpMap);
        pNewMP->AddObservation(pKFini, i);
        pKFini->AddMapPoint(pNewMP, i);
        pNewMP->ComputeDistinctiveDescriptors();
        pNewMP->UpdateNormalAndDepth();
        mpMap->AddMapPoint(pNewMP);

        if (FLAGS_ivslam_propagate_keyptqual) {
          pNewMP->SetQualityScore(mCurrentFrame.mvKeyQualScore[i]);
        }

        mCurrentFrame.mvpMapPoints[i] = pNewMP;
      }
    }

    //         cout << "New map created with " << mpMap->MapPointsInMap()
    //                 << " points" << endl;

    mpLocalMapper->InsertKeyFrame(pKFini);
    if (mbSingleThreaded) {
      mpLocalMapper->LoopOnce();
    }

    mLastFrame = Frame(mCurrentFrame);
    mnLastKeyFrameId = mCurrentFrame.mnId;
    mpLastKeyFrame = pKFini;

    mvpLocalKeyFrames.push_back(pKFini);
    mvpLocalMapPoints = mpMap->GetAllMapPoints();
    mpReferenceKF = pKFini;
    mCurrentFrame.mpReferenceKF = pKFini;

    mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

    mpMap->mvpKeyFrameOrigins.push_back(pKFini);

    if (mbTrainingMode) {
      mpMapDrawer->SetCurrentCameraPosewithGT(mCurrentFrame.mTcw,
                                              mCurrentFrame.mTwc_gt,
                                              mCurrentFrame.mstrLeftImgName);
    } else {
      mpMapDrawer->SetCurrentCameraPose(mCurrentFrame.mTcw,
                                        mCurrentFrame.mstrLeftImgName);
    }

    mState = OK;
  }
}

void Tracking::MonocularInitialization() {
  if (!mpInitializer) {
    // Set Reference Frame
    if (mCurrentFrame.mvKeys.size() > 100) {
      mInitialFrame = Frame(mCurrentFrame);
      mLastFrame = Frame(mCurrentFrame);
      mvbPrevMatched.resize(mCurrentFrame.mvKeysUn.size());
      for (size_t i = 0; i < mCurrentFrame.mvKeysUn.size(); i++)
        mvbPrevMatched[i] = mCurrentFrame.mvKeysUn[i].pt;

      if (mpInitializer) delete mpInitializer;

      mpInitializer = new Initializer(mCurrentFrame, 1.0, 200);

      fill(mvIniMatches.begin(), mvIniMatches.end(), -1);

      return;
    }
  } else {
    // Try to initialize
    if ((int)mCurrentFrame.mvKeys.size() <= 100) {
      delete mpInitializer;
      mpInitializer = static_cast<Initializer*>(NULL);
      fill(mvIniMatches.begin(), mvIniMatches.end(), -1);
      return;
    }

    // Find correspondences
    ORBmatcher matcher(0.9, true);
    int nmatches = matcher.SearchForInitialization(
        mInitialFrame, mCurrentFrame, mvbPrevMatched, mvIniMatches, 100);

    // Check if there are enough correspondences
    if (nmatches < 100) {
      delete mpInitializer;
      mpInitializer = static_cast<Initializer*>(NULL);
      return;
    }

    cv::Mat Rcw;                  // Current Camera Rotation
    cv::Mat tcw;                  // Current Camera Translation
    vector<bool> vbTriangulated;  // Triangulated Correspondences (mvIniMatches)

    if (mpInitializer->Initialize(
            mCurrentFrame, mvIniMatches, Rcw, tcw, mvIniP3D, vbTriangulated)) {
      for (size_t i = 0, iend = mvIniMatches.size(); i < iend; i++) {
        if (mvIniMatches[i] >= 0 && !vbTriangulated[i]) {
          mvIniMatches[i] = -1;
          nmatches--;
        }
      }

      // Set Frame Poses
      if (mbTrainingMode && mbGuidedBA) {
        mCurrentFrame.ApplyReferencePose();
      } else {
        mInitialFrame.SetPose(cv::Mat::eye(4, 4, CV_32F));
      }

      // TODO: In mbGuidedBA mode, apply Rcw and tcw to mCurrentFrame.Tcw
      if (mbTrainingMode && mbGuidedBA) {
        LOG(FATAL) << "mbGuidedBA is not yet supported for Monocular mode!";
      }

      cv::Mat Tcw = cv::Mat::eye(4, 4, CV_32F);
      Rcw.copyTo(Tcw.rowRange(0, 3).colRange(0, 3));
      tcw.copyTo(Tcw.rowRange(0, 3).col(3));
      mCurrentFrame.SetPose(Tcw);

      CreateInitialMapMonocular();
    }
  }
}

void Tracking::CreateInitialMapMonocular() {
  // Create KeyFrames
  KeyFrame* pKFini = new KeyFrame(mInitialFrame, mpMap, mpKeyFrameDB);
  KeyFrame* pKFcur = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);

  pKFini->ComputeBoW();
  pKFcur->ComputeBoW();

  // Insert KFs in the map
  mpMap->AddKeyFrame(pKFini);
  mpMap->AddKeyFrame(pKFcur);

  // Create MapPoints and asscoiate to keyframes
  for (size_t i = 0; i < mvIniMatches.size(); i++) {
    if (mvIniMatches[i] < 0) continue;

    // Create MapPoint.
    cv::Mat worldPos(mvIniP3D[i]);

    MapPoint* pMP = new MapPoint(worldPos, pKFcur, mpMap);

    pKFini->AddMapPoint(pMP, i);
    pKFcur->AddMapPoint(pMP, mvIniMatches[i]);

    pMP->AddObservation(pKFini, i);
    pMP->AddObservation(pKFcur, mvIniMatches[i]);

    pMP->ComputeDistinctiveDescriptors();
    pMP->UpdateNormalAndDepth();

    // Fill Current Frame structure
    mCurrentFrame.mvpMapPoints[mvIniMatches[i]] = pMP;
    mCurrentFrame.mvbOutlier[mvIniMatches[i]] = false;

    // Add to Map
    mpMap->AddMapPoint(pMP);

    if (FLAGS_ivslam_propagate_keyptqual) {
      float qual_score =
          std::min(mInitialFrame.mvKeyQualScore[i],
                   mCurrentFrame.mvKeyQualScore[mvIniMatches[i]]);
      pMP->SetQualityScore(qual_score);
    }
  }

  // Update Connections
  pKFini->UpdateConnections();
  pKFcur->UpdateConnections();

  // Bundle Adjustment
  cout << "New Map created with " << mpMap->MapPointsInMap() << " points"
       << endl;

  Optimizer::GlobalBundleAdjustemnt(mpMap, 20);

  // Set median depth to 1
  float medianDepth = pKFini->ComputeSceneMedianDepth(2);
  float invMedianDepth = 1.0f / medianDepth;

  // Original TrackedMapPoints threshold: 100
  // Relaxed TrackedMapPoints threshold: 20
  if (medianDepth < 0 || pKFcur->TrackedMapPoints(1) < 20) {
    cout << "Wrong initialization, reseting..." << endl;
    Reset();
    return;
  }

  // Scale initial baseline
  cv::Mat Tc2w = pKFcur->GetPose();
  Tc2w.col(3).rowRange(0, 3) = Tc2w.col(3).rowRange(0, 3) * invMedianDepth;
  pKFcur->SetPose(Tc2w);

  // Scale points
  vector<MapPoint*> vpAllMapPoints = pKFini->GetMapPointMatches();
  for (size_t iMP = 0; iMP < vpAllMapPoints.size(); iMP++) {
    if (vpAllMapPoints[iMP]) {
      MapPoint* pMP = vpAllMapPoints[iMP];
      pMP->SetWorldPos(pMP->GetWorldPos() * invMedianDepth);
    }
  }

  mpLocalMapper->InsertKeyFrame(pKFini);
  mpLocalMapper->InsertKeyFrame(pKFcur);
  if (mbSingleThreaded) {
    mpLocalMapper->LoopOnce();
    mpLocalMapper->LoopOnce();
  }

  mCurrentFrame.SetPose(pKFcur->GetPose());
  mnLastKeyFrameId = mCurrentFrame.mnId;
  mpLastKeyFrame = pKFcur;

  mvpLocalKeyFrames.push_back(pKFcur);
  mvpLocalKeyFrames.push_back(pKFini);
  mvpLocalMapPoints = mpMap->GetAllMapPoints();
  mpReferenceKF = pKFcur;
  mCurrentFrame.mpReferenceKF = pKFcur;

  mLastFrame = Frame(mCurrentFrame);

  mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

  if (mbTrainingMode) {
    mpMapDrawer->SetCurrentCameraPosewithGT(pKFcur->GetPose(),
                                            mCurrentFrame.mTwc_gt,
                                            mCurrentFrame.mstrLeftImgName);
  } else {
    mpMapDrawer->SetCurrentCameraPose(pKFcur->GetPose(),
                                      mCurrentFrame.mstrLeftImgName);
  }

  mpMap->mvpKeyFrameOrigins.push_back(pKFini);

  mState = OK;
}

void Tracking::CheckReplacedInLastFrame() {
  for (int i = 0; i < mLastFrame.N; i++) {
    MapPoint* pMP = mLastFrame.mvpMapPoints[i];

    if (pMP) {
      MapPoint* pRep = pMP->GetReplaced();
      if (pRep) {
        mLastFrame.mvpMapPoints[i] = pRep;
      }
    }
  }
}

bool Tracking::TrackReferenceKeyFrame(float nn_ratio_mult,
                                      float search_wind_mult,
                                      bool use_BoW) {
  // We perform first an ORB matching with the reference keyframe
  // If enough matches are found we setup a PnP solver
  ORBmatcher matcher(nn_ratio_mult * 0.7, true);
  vector<MapPoint*> vpMapPointMatches;
  mCurrentFrame.SetPose(mLastFrame.mTcw);
  int nmatches = 0;

  // Use Search by BOW for finding matches with map points visible by
  // the reference keyframe
  if (use_BoW) {
    // Compute Bag of Words vector
    mCurrentFrame.ComputeBoW();
    nmatches =
        matcher.SearchByBoW(mpReferenceKF, mCurrentFrame, vpMapPointMatches);

    if (nmatches < 15) return false;
    mCurrentFrame.mvpMapPoints = vpMapPointMatches;
    if (FLAGS_ivslam_propagate_keyptqual) {
      matcher.UpdateQualityScores(mCurrentFrame);
    }
  } else {
    // Use Search by projection to find matches with local map points

    fill(mCurrentFrame.mvpMapPoints.begin(),
         mCurrentFrame.mvpMapPoints.end(),
         static_cast<MapPoint*>(NULL));

    int nToMatch = 0;

    // Project points in frame and check its visibility
    for (vector<MapPoint*>::iterator vit = mvpLocalMapPoints.begin(),
                                     vend = mvpLocalMapPoints.end();
         vit != vend;
         vit++) {
      MapPoint* pMP = *vit;
      if (pMP->mnLastFrameSeen == mCurrentFrame.mnId) continue;
      if (pMP->isBad()) continue;
      // Project (this fills MapPoint variables for matching)
      if (mCurrentFrame.isInFrustum(pMP, 0.5)) {
        nToMatch++;
      }
    }

    if (nToMatch > 0) {
      int th = 5;
      nmatches = matcher.SearchByProjection(
          mCurrentFrame, mvpLocalMapPoints, search_wind_mult * th);
    }
    if (nmatches < 15) return false;
  }

  // Do additional logging if in training mode
  Optimizer::PoseOptimization(&mCurrentFrame, mbUnsupervisedLearning);
  if (mbGuidedBA) {
    mCurrentFrame.ApplyReferencePose();
  }
  if (!mbIntrospectionOn && mbTrainingMode) {
    mCurrentFrame.BackupNewMapPoints();
  }

  // Discard outliers
  int nmatchesMap = 0;
  for (int i = 0; i < mCurrentFrame.N; i++) {
    if (mCurrentFrame.mvpMapPoints[i]) {
      if (mCurrentFrame.mvbOutlier[i]) {
        MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

        // TODO: Remove this testing: +++
        //                 pMP->SetQualityScore(-1.0);
        //                 mCurrentFrame.mvKeyQualScore[i] = 0;
        // ---

        mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
        mCurrentFrame.mvbOutlier[i] = false;
        pMP->mbTrackInView = false;
        pMP->mnLastFrameSeen = mCurrentFrame.mnId;
        nmatches--;
      } else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
        nmatchesMap++;
    }
  }

  return nmatchesMap >= 10;
}

void Tracking::UpdateLastFrame() {
  // Update pose according to reference keyframe
  KeyFrame* pRef = mLastFrame.mpReferenceKF;
  if (!pRef) return;

  if (mlRelativeFramePoses.empty()) {
    return;
  }
  cv::Mat Tlr = mlRelativeFramePoses.back();
  mLastFrame.SetPose(Tlr * pRef->GetPose());

  if (mnLastKeyFrameId == mLastFrame.mnId || mSensor == System::MONOCULAR ||
      !mbOnlyTracking)
    return;

  // Create "visual odometry" MapPoints
  // We sort points according to their measured depth by the stereo/RGB-D sensor
  vector<pair<float, int>> vDepthIdx;
  vDepthIdx.reserve(mLastFrame.N);
  for (int i = 0; i < mLastFrame.N; i++) {
    float z = mLastFrame.mvDepth[i];
    if (z > 0) {
      vDepthIdx.push_back(make_pair(z, i));
    }
  }

  if (vDepthIdx.empty()) return;

  sort(vDepthIdx.begin(), vDepthIdx.end());

  // We insert all close points (depth<mThDepth)
  // If less than 100 close points, we insert the 100 closest ones.
  int nPoints = 0;
  for (size_t j = 0; j < vDepthIdx.size(); j++) {
    int i = vDepthIdx[j].second;

    bool bCreateNew = false;

    MapPoint* pMP = mLastFrame.mvpMapPoints[i];
    if (!pMP)
      bCreateNew = true;
    else if (pMP->Observations() < 1) {
      bCreateNew = true;
    }

    if (bCreateNew) {
      cv::Mat x3D = mLastFrame.UnprojectStereo(i);
      MapPoint* pNewMP = new MapPoint(x3D, mpMap, &mLastFrame, i);

      mLastFrame.mvpMapPoints[i] = pNewMP;

      mlpTemporalPoints.push_back(pNewMP);
      nPoints++;
    } else {
      nPoints++;
    }

    if (vDepthIdx[j].first > mThDepth && nPoints > 100) break;
  }
}

bool Tracking::TrackWithMotionModel(float nn_ratio_mult,
                                    float search_wind_mult) {
  ORBmatcher matcher(nn_ratio_mult * mMatcherNNRatioMultiplier * 0.9, true);

  // Update last frame pose according to its reference keyframe
  // Create "visual odometry" points if in Localization Mode
  UpdateLastFrame();

  if (!mVelocity.empty()) {
    mCurrentFrame.SetPose(mVelocity * mLastFrame.mTcw);
  } else {
    mCurrentFrame.SetPose(mLastFrame.mTcw);
  }

  fill(mCurrentFrame.mvpMapPoints.begin(),
       mCurrentFrame.mvpMapPoints.end(),
       static_cast<MapPoint*>(NULL));

  // Project points seen in previous frame
  int th;
  if (mSensor != System::STEREO)
    th = 15;
  else
    th = 7;
  int nmatches = matcher.SearchByProjection(
      mCurrentFrame,
      mLastFrame,
      search_wind_mult * mMatcherSearchWindowMultiplier * th,
      mSensor == System::MONOCULAR);

  // If few matches, uses a wider window search
  if (nmatches < 20) {
    fill(mCurrentFrame.mvpMapPoints.begin(),
         mCurrentFrame.mvpMapPoints.end(),
         static_cast<MapPoint*>(NULL));
    nmatches = matcher.SearchByProjection(
        mCurrentFrame,
        mLastFrame,
        search_wind_mult * mMatcherSearchWindowMultiplier * 2 * th,
        mSensor == System::MONOCULAR);
  }

  //     if (search_wind_mult > 1.0) {
  //       std::cout << "nmatches: " << nmatches << std::endl;
  //     }

  if (nmatches < 20) return false;

  // +++++++++++++++++++++++++++++++++++++++++++++++++
  // +++++++++++++++++++++++++++++++++++++++++++++++++
  // TODO: Remove this temporary sanity checking:
  // Evaluate the quality of keypoints/map points given the ground truth
  // depth and hence ground truth reprojection error.
  const bool kEnableKeyPointEval = false;
  const bool kUseEpipolarErr = true;
  const float kReprojErrMaxClamp = 10.0;  // 10.0
  const float kMinQualityScore = 0.5;
  const double kErrMinClamp = 0.0;
  const double kErrMaxClamp = 1.5;  // -7
  const bool kUseAnalyticalUncertaintyPropagation = true;
  if (kEnableKeyPointEval) {
    for (size_t i = 0; i < mCurrentFrame.mvKeysUn.size(); i++) {
      // Add the keypoints that are matched with map points
      MapPoint* curr_map_pt = mCurrentFrame.mvpMapPoints[i];
      if (curr_map_pt) {
        cv::KeyPoint keypt_curr = mCurrentFrame.mvKeysUn[i];
        cv::KeyPoint keypt_prev;
        int keypt_prev_idx;

        bool prev_keypt_available = mLastFrame.GetCorrespondingKeyPt(
            curr_map_pt, &keypt_prev, &keypt_prev_idx);
        if (!prev_keypt_available) {
          LOG(WARNING) << "Could not find a matching keypoint in "
                       << "prev frame!";
          continue;
        }

        cv::Point2f reproj_gt;
        double scaled_err = 0;
        bool gt_depth_available = mFeatureEvaluator->GetGTReprojection(
            mLastFrame, mCurrentFrame, keypt_prev_idx, i, &reproj_gt);

        if (kUseEpipolarErr) {
          Eigen::Vector2f epipolar_line_dir;
          Eigen::Vector2f proj_on_epipolar_line;
          vector<Eigen::Vector2f> sigma_pts_err;
          Eigen::Matrix2f epipolar_err_covariance;
          float epipolar_err_var;
          double err_norm_factor;
          double err;

          if (kUseAnalyticalUncertaintyPropagation) {
            err = mFeatureEvaluator->CalculateNormalizedEpipolarErrorAnalytical(
                mLastFrame.mTwc_gt,
                mLastFrame.mSigmaTwc_gt,
                mLastFrame.mbPoseUncertaintyAvailable,
                mLastFrame.mstrLeftImgName,
                mCurrentFrame,
                keypt_prev,
                keypt_curr,
                &epipolar_line_dir,
                &proj_on_epipolar_line,
                &epipolar_err_var,
                &err_norm_factor);

          } else {
            err = mFeatureEvaluator->CalculateNormalizedEpipolarError(
                mLastFrame.mTwc_gt,
                mCurrentFrame,
                keypt_prev,
                keypt_curr,
                &epipolar_line_dir,
                &proj_on_epipolar_line,
                &sigma_pts_err,
                &epipolar_err_covariance,
                &err_norm_factor);
          }

          //             double err_log = log(err);
          double err_log = err;
          scaled_err = static_cast<double>((err_log - kErrMinClamp) /
                                           (kErrMaxClamp - kErrMinClamp));
          scaled_err = (scaled_err > 0.9) ? 0.9 : scaled_err;
          scaled_err = (scaled_err < 0.0) ? 0.0 : scaled_err;
        } else if (gt_depth_available) {
          Eigen::Vector2f reproj_err(reproj_gt.x - keypt_curr.pt.x,
                                     reproj_gt.y - keypt_curr.pt.y);
          float err_abs = sqrt(reproj_err.x() * reproj_err.x() +
                               reproj_err.y() * reproj_err.y());
          scaled_err = err_abs / kReprojErrMaxClamp;
          //           cout << "reproj err vec: " << reproj_err.transpose() <<
          //           endl; cout << "scaled err: " << scaled_err << endl;
          scaled_err = (scaled_err > 1.0) ? 1.0 : scaled_err;
          scaled_err = (scaled_err < 0.0) ? 0.0 : scaled_err;
        }

        float qual_score = 1.0 / (1.0 + scaled_err);
        float qual_score_norm = 2 * qual_score - 1;

        curr_map_pt->SetQualityScore(qual_score_norm);

        //         if(qual_score_norm < kMinQualityScore) {
        //           LOG(WARNING) << "Removing Pose Optim map point: " <<
        //                            qual_score_norm;
        //           curr_map_pt->SetBadFlag();
        //         }

        if (!isnan(qual_score_norm)) {
          mCurrentFrame.mvKeyQualScore[i] = qual_score_norm;
        }
      }
    }
  }
  // ---------------------------------------------------
  // ---------------------------------------------------

  // Optimize frame pose with all matches
  // Do additional logging if in training mode
  Optimizer::PoseOptimization(&mCurrentFrame, mbUnsupervisedLearning);
  if (mbGuidedBA) {
    // Overwrites the result of optimization
    mCurrentFrame.ApplyReferencePose();
  }

  if (!mbIntrospectionOn && mbTrainingMode) {
    mCurrentFrame.BackupNewMapPoints();
  }

  // Discard outliers
  int nmatchesMap = 0;
  for (int i = 0; i < mCurrentFrame.N; i++) {
    if (mCurrentFrame.mvpMapPoints[i]) {
      // TODO : remove this temporary test
      //           nmatchesMap++;
      // ------

      if (mCurrentFrame.mvbOutlier[i]) {
        MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];

        // TODO: Remove this testing: +++
        //                 pMP->SetQualityScore(-1.0);
        //                 mCurrentFrame.mvKeyQualScore[i] = 0;
        // ---

        mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
        mCurrentFrame.mvbOutlier[i] = false;
        pMP->mbTrackInView = false;
        pMP->mnLastFrameSeen = mCurrentFrame.mnId;
        nmatches--;
      } else if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
        nmatchesMap++;
    }
  }

  //     if (search_wind_mult > 1.0) {
  //       std::cout << "nmatches pruned: " << nmatches << std::endl;
  //     }

  if (mbOnlyTracking) {
    mbVO = nmatchesMap < 10;
    return nmatches > 20;
  }

  return nmatchesMap >= 10;
}

bool Tracking::TrackLocalMap() {
  // We have an estimation of the camera pose and some map points tracked in the
  // frame. We retrieve the local map and try to find matches to points in the
  // local map.

  UpdateLocalMap();

  SearchLocalPoints();

  // +++++++++++++++++++++++++++++++++++++++++++++++++
  // +++++++++++++++++++++++++++++++++++++++++++++++++
  // TODO: Remove this temporary sanity checking#2:
  // Evaluate the quality of keypoints/map points given the ground truth
  // depth and hence ground truth reprojection error.
  const bool kEnableKeyPointEval = false;
  const bool kUseEpipolarErr = true;
  const float kReprojErrMaxClamp = 10.0;  // 10.0
  const float kMinQualityScore = 0.5;
  const double kErrMinClamp = 0.0;
  const double kErrMaxClamp = 1.5;  // -7
  const bool kUseAnalyticalUncertaintyPropagation = true;
  if (kEnableKeyPointEval) {
    for (size_t i = 0; i < mCurrentFrame.mvKeysUn.size(); i++) {
      // Add the keypoints that are matched with map points
      MapPoint* curr_map_pt = mCurrentFrame.mvpMapPoints[i];
      if (curr_map_pt) {
        cv::KeyPoint keypt_curr = mCurrentFrame.mvKeysUn[i];
        cv::KeyPoint keypt_prev;

        // Reference frame of current map point
        KeyFrame* pt_ref_keyframe = curr_map_pt->GetReferenceKeyFrame();
        //       cout << "Ref frame ID: " << pt_ref_keyframe->mnId << endl;
        //       cout << "curr frame ID: " << curr_frame.mnId << endl;
        int pt_idx_in_frame = curr_map_pt->GetIndexInKeyFrame(pt_ref_keyframe);

        if (pt_idx_in_frame < 0) {
          LOG(WARNING) << "Could not find a matching keypoint in "
                       << "reference frame!";
          continue;
        }

        keypt_prev = pt_ref_keyframe->mvKeysUn[pt_idx_in_frame];
        cv::Point2f reproj_gt;
        bool uncertain_gt_depth;
        double scaled_err = 0;
        bool gt_depth_available =
            mFeatureEvaluator->GetGTReprojection(*pt_ref_keyframe,
                                                 mCurrentFrame,
                                                 pt_idx_in_frame,
                                                 i,
                                                 &reproj_gt,
                                                 &uncertain_gt_depth);

        if (kUseEpipolarErr) {
          Eigen::Vector2f epipolar_line_dir;
          Eigen::Vector2f proj_on_epipolar_line;
          vector<Eigen::Vector2f> sigma_pts_err;
          Eigen::Matrix2f epipolar_err_covariance;
          float epipolar_err_var;
          double err_norm_factor;
          double err;

          if (kUseAnalyticalUncertaintyPropagation) {
            err = mFeatureEvaluator->CalculateNormalizedEpipolarErrorAnalytical(
                pt_ref_keyframe->mTwc_gt,
                pt_ref_keyframe->mSigmaTwc_gt,
                pt_ref_keyframe->mbPoseUncertaintyAvailable,
                pt_ref_keyframe->mstrLeftImgName,
                mCurrentFrame,
                keypt_prev,
                keypt_curr,
                &epipolar_line_dir,
                &proj_on_epipolar_line,
                &epipolar_err_var,
                &err_norm_factor);

          } else {
            err = mFeatureEvaluator->CalculateNormalizedEpipolarError(
                pt_ref_keyframe->mTwc_gt,
                mCurrentFrame,
                keypt_prev,
                keypt_curr,
                &epipolar_line_dir,
                &proj_on_epipolar_line,
                &sigma_pts_err,
                &epipolar_err_covariance,
                &err_norm_factor);
          }

          //             double err_log = log(err);
          double err_log = err;
          scaled_err = static_cast<double>((err_log - kErrMinClamp) /
                                           (kErrMaxClamp - kErrMinClamp));
          scaled_err = (scaled_err > 0.9) ? 0.9 : scaled_err;
          scaled_err = (scaled_err < 0.0) ? 0.0 : scaled_err;
        } else if (gt_depth_available && !uncertain_gt_depth) {
          Eigen::Vector2f reproj_err(reproj_gt.x - keypt_curr.pt.x,
                                     reproj_gt.y - keypt_curr.pt.y);
          float err_abs = sqrt(reproj_err.x() * reproj_err.x() +
                               reproj_err.y() * reproj_err.y());
          scaled_err = err_abs / kReprojErrMaxClamp;
          //           cout << "reproj err vec: " << reproj_err.transpose() <<
          //           endl; cout << "scaled err: " << scaled_err << endl;
          scaled_err = (scaled_err > 1.0) ? 1.0 : scaled_err;
          scaled_err = (scaled_err < 0.0) ? 0.0 : scaled_err;
        }

        float qual_score = 1.0 / (1.0 + scaled_err);
        float qual_score_norm = 2 * qual_score - 1;

        curr_map_pt->SetQualityScore(qual_score_norm);

        //         if(qual_score_norm < kMinQualityScore) {
        //           LOG(WARNING) << "Removing Pose Optim map point: " <<
        //                            qual_score_norm;
        //           curr_map_pt->SetBadFlag();
        //         }

        if (!isnan(qual_score_norm)) {
          mCurrentFrame.mvKeyQualScore[i] = qual_score_norm;
        }
      }
    }
  }
  // ---------------------------------------------------
  // ---------------------------------------------------

  // Optimize Pose
  // Do additional logging if in training mode
  Optimizer::PoseOptimization(&mCurrentFrame, mbUnsupervisedLearning);
  if (mbGuidedBA) {
    // Overwrites the result of optimization
    mCurrentFrame.ApplyReferencePose();
  }

  if (!mbIntrospectionOn && mbTrainingMode) {
    mCurrentFrame.BackupNewMapPoints();
  }

  mnMatchesInliers = 0;

  // Update MapPoints Statistics
  for (int i = 0; i < mCurrentFrame.N; i++) {
    if (mCurrentFrame.mvpMapPoints[i]) {
      // TODO : remove this temporary test
      //           mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
      //           if(mCurrentFrame.mvpMapPoints[i]->Observations()>0) {
      //                     mnMatchesInliers++;
      //           }
      // ------

      if (!mCurrentFrame.mvbOutlier[i]) {
        mCurrentFrame.mvpMapPoints[i]->IncreaseFound();
        if (!mbOnlyTracking) {
          if (mCurrentFrame.mvpMapPoints[i]->Observations() > 0)
            mnMatchesInliers++;
        } else
          mnMatchesInliers++;
      } else {
        // TODO: Remove this testing: +++
        //               mCurrentFrame.mvpMapPoints[i]->SetQualityScore(-1.0);
        //               mCurrentFrame.mvKeyQualScore[i] = 0;
        // ---

        if (mSensor == System::STEREO)
          mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
      }
    }
  }

  //     std::cout << "TrackLocal map inliers: " << mnMatchesInliers <<
  //     std::endl;

  // Decide if the tracking was succesful
  // More restrictive if there was a relocalization recently
  if (mCurrentFrame.mnId < mnLastRelocFrameId + mMaxFrames &&
      mnMatchesInliers < 50)
    return false;

  if (mnMatchesInliers < 30)
    return false;
  else
    return true;
}

bool Tracking::NeedNewKeyFrame() {
  if (mbOnlyTracking) return false;

  // If Local Mapping is freezed by a Loop Closure do not insert keyframes
  if (mpLocalMapper->isStopped() || mpLocalMapper->stopRequested())
    return false;

  const int nKFs = mpMap->KeyFramesInMap();

  // Do not insert keyframes if not enough frames have passed from last
  // relocalisation
  if (mCurrentFrame.mnId < mnLastRelocFrameId + mMaxFrames && nKFs > mMaxFrames)
    return false;

  // Tracked MapPoints in the reference keyframe
  int nMinObs = 3;
  if (nKFs <= 2) nMinObs = 2;
  int nRefMatches = mpReferenceKF->TrackedMapPoints(nMinObs);

  // Local Mapping accept keyframes?
  bool bLocalMappingIdle = mpLocalMapper->AcceptKeyFrames();

  // Check how many "close" points are being tracked and how many could be
  // potentially created.
  int nNonTrackedClose = 0;
  int nTrackedClose = 0;
  if (mSensor != System::MONOCULAR) {
    for (int i = 0; i < mCurrentFrame.N; i++) {
      if (mCurrentFrame.mvDepth[i] > 0 && mCurrentFrame.mvDepth[i] < mThDepth) {
        if (mCurrentFrame.mvpMapPoints[i] && !mCurrentFrame.mvbOutlier[i])
          nTrackedClose++;
        else
          nNonTrackedClose++;
      }
    }
  }

  bool bNeedToInsertClose = (nTrackedClose < 100) && (nNonTrackedClose > 70);

  // Thresholds
  float thRefRatio = 0.75f;
  if (nKFs < 2) thRefRatio = 0.4f;

  if (mSensor == System::MONOCULAR) thRefRatio = 0.9f;

  // Condition 1a: More than "MaxFrames" have passed from last keyframe
  // insertion
  const bool c1a = mCurrentFrame.mnId >= mnLastKeyFrameId + mMaxFrames;
  // Condition 1b: More than "MinFrames" have passed and Local Mapping is idle
  const bool c1b = (mCurrentFrame.mnId >= mnLastKeyFrameId + mMinFrames &&
                    bLocalMappingIdle);
  // Condition 1c: tracking is weak
  const bool c1c =
      mSensor != System::MONOCULAR &&
      (mnMatchesInliers < nRefMatches * 0.25 || bNeedToInsertClose);
  // Condition 2: Few tracked points compared to reference keyframe. Lots of
  // visual odometry compared to map matches.
  const bool c2 =
      ((mnMatchesInliers < nRefMatches * thRefRatio || bNeedToInsertClose) &&
       mnMatchesInliers > 15);

  if ((c1a || c1b || c1c) && c2) {
    // If the mapping accepts keyframes, insert keyframe.
    // Otherwise send a signal to interrupt BA
    if (bLocalMappingIdle) {
      return true;
    } else {
      mpLocalMapper->InterruptBA();
      if (mSensor != System::MONOCULAR) {
        if (mpLocalMapper->KeyframesInQueue() < 3)
          return true;
        else
          return false;
      } else
        return false;
    }
  } else
    return false;
}

void Tracking::CreateNewKeyFrame() {
  if (!mpLocalMapper->SetNotStop(true)) return;

  KeyFrame* pKF = new KeyFrame(mCurrentFrame, mpMap, mpKeyFrameDB);

  mpReferenceKF = pKF;
  mCurrentFrame.mpReferenceKF = pKF;

  if (mSensor != System::MONOCULAR) {
    mCurrentFrame.UpdatePoseMatrices();

    // We sort points by the measured depth by the stereo/RGBD sensor.
    // We create all those MapPoints whose depth < mThDepth.
    // If there are less than 100 close points we create the 100 closest.
    vector<pair<float, int>> vDepthIdx;
    vDepthIdx.reserve(mCurrentFrame.N);
    for (int i = 0; i < mCurrentFrame.N; i++) {
      float z = mCurrentFrame.mvDepth[i];
      if (z > 0) {
        vDepthIdx.push_back(make_pair(z, i));
      }
    }

    if (!vDepthIdx.empty()) {
      sort(vDepthIdx.begin(), vDepthIdx.end());

      int nPoints = 0;
      for (size_t j = 0; j < vDepthIdx.size(); j++) {
        int i = vDepthIdx[j].second;

        bool bCreateNew = false;

        MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
        if (!pMP)
          bCreateNew = true;
        else if (pMP->Observations() < 1) {
          bCreateNew = true;
          mCurrentFrame.mvpMapPoints[i] = static_cast<MapPoint*>(NULL);
        }

        if (bCreateNew) {
          cv::Mat x3D = mCurrentFrame.UnprojectStereo(i);
          MapPoint* pNewMP = new MapPoint(x3D, pKF, mpMap);
          pNewMP->AddObservation(pKF, i);
          pKF->AddMapPoint(pNewMP, i);
          pNewMP->ComputeDistinctiveDescriptors();
          pNewMP->UpdateNormalAndDepth();
          mpMap->AddMapPoint(pNewMP);

          if (FLAGS_ivslam_propagate_keyptqual) {
            pNewMP->SetQualityScore(mCurrentFrame.mvKeyQualScore[i]);
          }

          mCurrentFrame.mvpMapPoints[i] = pNewMP;
          nPoints++;
        } else {
          nPoints++;
        }

        if (vDepthIdx[j].first > mThDepth && nPoints > 100) break;
      }
    }
  }

  mpLocalMapper->InsertKeyFrame(pKF);
  if (mbSingleThreaded) {
    bool run_ba = false;
    if (mFramesReceivedSinceLastLocalBA > FLAGS_tracking_ba_rate ||
        mpMap->KeyFramesInMap() < 5) {
      run_ba = true;

      mFramesReceivedSinceLastLocalBA = 0;
    }

    mpLocalMapper->LoopOnce(run_ba);
  }

  mpLocalMapper->SetNotStop(false);

  mnLastKeyFrameId = mCurrentFrame.mnId;
  mpLastKeyFrame = pKF;
}

void Tracking::VisualizeKeyPtsQuality() {
  // If set to true, the overall quality score of the corresponding map point
  // will be used
  // when visualizing the quality of each keypoint. If not, the direct quality
  // score that is stored for that keypoint will be used.
  const bool visualize_map_pt_qual = true;
  const int kMinObsNum = 3;

  const int radius = 3;  // 5
  const float radius_range = 8;
  const float min_obs = 3;
  const float max_obs = 13;

  const string src_dir =
      "/media/ssd2/datasets/Jackal_Visual_Odom/"
      "sequences/00042/image_0/";

  const string out_dir =
      "/media/ssd2/results/introspective_SLAM/feature_evaluation/"
      "Jackal_Visual_Odom/"
      "ORB_SLAM2_epipolar_test/Normalized_CompWithRefFrame_test/"
      "00042/";

  // Load corresponding image from file

  //   KeyFrame* key_frame = mpLastKeyFrame;
  KeyFrame* key_frame;
  long unsigned int lastKF_id = mpLastKeyFrame->mnId;
  long unsigned int interesKF_id = lastKF_id - 6;

  bool found = false;
  for (size_t i = 0; i < mvpLocalKeyFrames.size(); i++) {
    if (mvpLocalKeyFrames[i]->mnId == interesKF_id) {
      key_frame = mvpLocalKeyFrames[i];
      found = true;
      break;
    }
  }

  if (!found) {
    LOG(WARNING) << "Previous keyframe not found";
    return;
  }

  string img_name = key_frame->mstrLeftImgName;
  cv::Mat img = cv::imread(src_dir + "/" + img_name, CV_LOAD_IMAGE_UNCHANGED);

  if (img.empty()) {
    LOG(FATAL) << "Could not load image " << src_dir + "/" + img_name;
  }
  cv::Mat img_color;
  cvtColor(img, img_color, cv::COLOR_GRAY2BGR);

  for (size_t i = 0; i < key_frame->mvKeysUn.size(); i++) {
    MapPoint* m_pt = key_frame->GetMapPoint(i);
    if (m_pt) {
      if (m_pt->isBad()) {
        continue;
      }
      int obs_num = m_pt->Observations();

      if (obs_num < kMinObsNum) {
        continue;
      }

      // Increase the radius of each keypoint proportional to the number of
      // times its corresponding map point has been observed
      int radius_add = static_cast<int>(radius_range * (obs_num - min_obs) /
                                        (max_obs - min_obs));
      radius_add = (radius_add > radius_range) ? radius_range : radius_add;
      radius_add = (radius_add < 0) ? 0 : radius_add;

      cv::Point point(static_cast<int>(key_frame->mvKeysUn[i].pt.x),
                      static_cast<int>(key_frame->mvKeysUn[i].pt.y));

      int scaled_err;
      if (visualize_map_pt_qual) {
        if (m_pt->mbQualityScoreCalculated) {
          scaled_err = static_cast<int>(255 * m_pt->GetQualityScore());
        } else {
          // Just set to something larger than 256 to be visualized differently
          // (if no map point quality score is available)
          scaled_err = 2 * 255;
        }
      } else {
        scaled_err = static_cast<int>(255 * key_frame->mvKeyQualScore[i]);
      }
      cv::Scalar color;
      if (scaled_err > 256) {
        // keypoints without quality scores: blue
        color = cv::Scalar(125, 0, 0);
      } else {
        color = cv::Scalar(0, scaled_err, 255 - scaled_err);
      }
      circle(img_color, point, radius + radius_add, color, -1, 8, 0);
    }
  }

  cv::imshow("keypt quality", img_color);
  cv::waitKey(0);
}

bool Tracking::EvaluateTrackingAccuracy() {
  // If set to true, tracking accuracy is flagged as reliable only if current
  // velocity is larger than some threshold. This is to prune out regions
  // where the camera is standing still hence, feature tracking is easy.
  const bool kAssertMinVelocity = true;

  // If set to true, frame time stamps are used for calculating the velocity
  // of the camera. Otherwise, del_T will be deduced from the provided FPS
  // of the images.
  const bool kUseTimeStamps = false;

  const float kMinAngVel = M_PI * 10.0f / 180.0f;  // rad/s
  const float kMinLinVel = 0.3;                    // m/s

  const long int min_horizon = 20;
  const long int max_horizon = 35;

  const double chi2_thresh = 12.59159;  // 95% percentile for chi2 of degree 6

  long unsigned int curr_id = mCurrentFrame.mnId;
  KeyFrame* ref_kf;

  bool ref_found = false;
  for (long int k = min_horizon; k < max_horizon; k++) {
    long unsigned int ref_id = static_cast<long unsigned int>(
        std::max(0l, static_cast<long int>(curr_id) - k));
    if (ref_id == 0) {
      return false;
    }

    if (ref_found) {
      break;
    }

    for (size_t i = 0; i < mvpLocalKeyFrames.size(); i++) {
      if (mvpLocalKeyFrames[i]->mnFrameId == ref_id) {
        ref_kf = mvpLocalKeyFrames[i];
        ref_found = true;
        break;
      }
    }
  }

  if (!ref_found) {
    LOG(INFO) << "No keyframe found with mnFrameId in the range "
              << static_cast<long int>(curr_id) - max_horizon << ", "
              << static_cast<long int>(curr_id) - min_horizon;
    return false;
  }

  //   cout << "curr_frame/found_ref:  " << curr_id << " / " <<
  //   ref_kf->mnFrameId
  //        << endl;

  cv::Mat pose_gt_0 = ref_kf->GetGTPose();
  cv::Mat pose_gt_1 = mCurrentFrame.mTwc_gt;

  cv::Mat pose_est_0 = ref_kf->GetPoseInverse();
  cv::Mat pose_est_1 = CalculateInverseTransform(mCurrentFrame.mTcw);

  Eigen::AngleAxisd aa_rot_err;
  Eigen::Vector3d t_err;
  mFeatureEvaluator->CalcRelativePoseError(
      pose_est_0, pose_est_1, pose_gt_0, pose_gt_1, &aa_rot_err, &t_err);
  //   cout << "t_err: " << t_err.transpose() << endl;
  //   cout << "aa_rot_err" << aa_rot_err.axis().transpose() << ": "
  //        << aa_rot_err.angle() << endl;

  Eigen::Matrix<double, 6, 1> pose_err;
  pose_err.topRows(3) = aa_rot_err.axis() * aa_rot_err.angle();
  pose_err.bottomRows(3) = t_err;

  Eigen::Matrix<double, 6, 6> cam_pose_cov_inv;
  cam_pose_cov_inv << FLAGS_ivslam_ref_pose_ang_var_inv, 0, 0, 0, 0, 0, 0,
      FLAGS_ivslam_ref_pose_ang_var_inv, 0, 0, 0, 0, 0, 0,
      FLAGS_ivslam_ref_pose_ang_var_inv, 0, 0, 0, 0, 0, 0,
      FLAGS_ivslam_ref_pose_trans_var_inv, 0, 0, 0, 0, 0, 0,
      FLAGS_ivslam_ref_pose_trans_var_inv, 0, 0, 0, 0, 0, 0,
      FLAGS_ivslam_ref_pose_trans_var_inv;

  double chi2 = pose_err.transpose() * cam_pose_cov_inv * pose_err;

  //   cout << "chi2: " << chi2 << " / " << chi2_thresh << ". ";

  // Calculate cameras velocity w.r.t the last keyframe
  bool min_vel_acheived = true;
  if (kAssertMinVelocity) {
    cv::Mat pose_gt_lastKF = mpLastKeyFrame->GetGTPose();
    cv::Mat rel_tf = mFeatureEvaluator->CalculateRelativeTransform(
        pose_gt_lastKF, pose_gt_1);
    float del_t;
    if (kUseTimeStamps) {
      del_t = static_cast<float>(mCurrentFrame.mTimeStamp -
                                 mpLastKeyFrame->mTimeStamp);
    } else {
      del_t =
          (1.0 / mMaxFrames) * (mCurrentFrame.mnId - mpLastKeyFrame->mnFrameId);
    }

    Eigen::Map<Eigen::Matrix<float, 4, 4, Eigen::RowMajor>> T_rel(
        rel_tf.ptr<float>(), rel_tf.rows, rel_tf.cols);
    Eigen::Vector3f t_rel = T_rel.topRightCorner(3, 1);
    Eigen::Matrix3f R_rel = T_rel.topLeftCorner(3, 3);
    Eigen::AngleAxisf aa_rel(R_rel);

    float lin_vel_abs = (t_rel / del_t).norm();
    float ang_vel_abs = aa_rel.angle() / del_t;

    //     cout << "lin_vel_abs: " << lin_vel_abs << endl;
    //     cout << "ang_vel_abs: " << ang_vel_abs << endl;

    if (lin_vel_abs < kMinLinVel && ang_vel_abs < kMinAngVel) {
      min_vel_acheived = false;
      //       cout << "Velocity does not reach the minimum for training data
      //       gen"
      //            << endl;
      return false;
    }
  }

  if (chi2 > chi2_thresh) {
    //     cout << "Unreliable!!!!" << endl;
    return false;
  } else {
    //     cout << "Reliable." << endl;
    return true;
  }
}

void Tracking::SearchLocalPoints() {
  // Do not search map points already matched
  for (vector<MapPoint*>::iterator vit = mCurrentFrame.mvpMapPoints.begin(),
                                   vend = mCurrentFrame.mvpMapPoints.end();
       vit != vend;
       vit++) {
    MapPoint* pMP = *vit;
    if (pMP) {
      if (pMP->isBad()) {
        *vit = static_cast<MapPoint*>(NULL);
      } else {
        pMP->IncreaseVisible();
        pMP->mnLastFrameSeen = mCurrentFrame.mnId;
        pMP->mbTrackInView = false;
      }
    }
  }

  int nToMatch = 0;

  // Project points in frame and check its visibility
  for (vector<MapPoint*>::iterator vit = mvpLocalMapPoints.begin(),
                                   vend = mvpLocalMapPoints.end();
       vit != vend;
       vit++) {
    MapPoint* pMP = *vit;
    if (pMP->mnLastFrameSeen == mCurrentFrame.mnId) continue;
    if (pMP->isBad()) continue;
    // Project (this fills MapPoint variables for matching)
    if (mCurrentFrame.isInFrustum(pMP, 0.5)) {
      pMP->IncreaseVisible();
      nToMatch++;
    }
  }

  if (nToMatch > 0) {
    ORBmatcher matcher(mMatcherNNRatioMultiplier * 0.8);
    int th = 1;
    if (mSensor == System::RGBD) th = 3;
    // If the camera has been relocalised recently, perform a coarser search
    if (mCurrentFrame.mnId < mnLastRelocFrameId + 2) th = 5;
    matcher.SearchByProjection(
        mCurrentFrame, mvpLocalMapPoints, mMatcherSearchWindowMultiplier * th);
  }
}

void Tracking::UpdateLocalMap() {
  // This is for visualization
  mpMap->SetReferenceMapPoints(mvpLocalMapPoints);

  // Update
  UpdateLocalKeyFrames();
  UpdateLocalPoints();
}

void Tracking::UpdateLocalPoints() {
  mvpLocalMapPoints.clear();

  for (vector<KeyFrame*>::const_iterator itKF = mvpLocalKeyFrames.begin(),
                                         itEndKF = mvpLocalKeyFrames.end();
       itKF != itEndKF;
       itKF++) {
    KeyFrame* pKF = *itKF;
    const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();

    for (vector<MapPoint*>::const_iterator itMP = vpMPs.begin(),
                                           itEndMP = vpMPs.end();
         itMP != itEndMP;
         itMP++) {
      MapPoint* pMP = *itMP;
      if (!pMP) continue;
      if (pMP->mnTrackReferenceForFrame == mCurrentFrame.mnId) continue;
      if (!pMP->isBad()) {
        mvpLocalMapPoints.push_back(pMP);
        pMP->mnTrackReferenceForFrame = mCurrentFrame.mnId;
      }
    }
  }
}

void Tracking::UpdateLocalKeyFrames() {
  // Each map point vote for the keyframes in which it has been observed
  map<KeyFrame*, int> keyframeCounter;
  for (int i = 0; i < mCurrentFrame.N; i++) {
    if (mCurrentFrame.mvpMapPoints[i]) {
      MapPoint* pMP = mCurrentFrame.mvpMapPoints[i];
      if (!pMP->isBad()) {
        const map<KeyFrame*, size_t> observations = pMP->GetObservations();
        for (map<KeyFrame*, size_t>::const_iterator it = observations.begin(),
                                                    itend = observations.end();
             it != itend;
             it++)
          keyframeCounter[it->first]++;
      } else {
        mCurrentFrame.mvpMapPoints[i] = NULL;
      }
    }
  }

  if (keyframeCounter.empty()) return;

  int max = 0;
  KeyFrame* pKFmax = static_cast<KeyFrame*>(NULL);

  mvpLocalKeyFrames.clear();
  mvpLocalKeyFrames.reserve(3 * keyframeCounter.size());

  // All keyframes that observe a map point are included in the local map. Also
  // check which keyframe shares most points
  for (map<KeyFrame*, int>::const_iterator it = keyframeCounter.begin(),
                                           itEnd = keyframeCounter.end();
       it != itEnd;
       it++) {
    KeyFrame* pKF = it->first;

    if (pKF->isBad()) continue;

    if (it->second > max) {
      max = it->second;
      pKFmax = pKF;
    }

    mvpLocalKeyFrames.push_back(it->first);
    pKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
  }

  // Include also some not-already-included keyframes that are neighbors to
  // already-included keyframes
  for (vector<KeyFrame*>::const_iterator itKF = mvpLocalKeyFrames.begin(),
                                         itEndKF = mvpLocalKeyFrames.end();
       itKF != itEndKF;
       itKF++) {
    // Limit the number of keyframes
    if (mvpLocalKeyFrames.size() > 80) break;

    KeyFrame* pKF = *itKF;

    const vector<KeyFrame*> vNeighs = pKF->GetBestCovisibilityKeyFrames(10);

    for (vector<KeyFrame*>::const_iterator itNeighKF = vNeighs.begin(),
                                           itEndNeighKF = vNeighs.end();
         itNeighKF != itEndNeighKF;
         itNeighKF++) {
      KeyFrame* pNeighKF = *itNeighKF;
      if (!pNeighKF->isBad()) {
        if (pNeighKF->mnTrackReferenceForFrame != mCurrentFrame.mnId) {
          mvpLocalKeyFrames.push_back(pNeighKF);
          pNeighKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
          break;
        }
      }
    }

    const set<KeyFrame*> spChilds = pKF->GetChilds();
    for (set<KeyFrame*>::const_iterator sit = spChilds.begin(),
                                        send = spChilds.end();
         sit != send;
         sit++) {
      KeyFrame* pChildKF = *sit;
      if (!pChildKF->isBad()) {
        if (pChildKF->mnTrackReferenceForFrame != mCurrentFrame.mnId) {
          mvpLocalKeyFrames.push_back(pChildKF);
          pChildKF->mnTrackReferenceForFrame = mCurrentFrame.mnId;
          break;
        }
      }
    }

    KeyFrame* pParent = pKF->GetParent();
    if (pParent) {
      if (pParent->mnTrackReferenceForFrame != mCurrentFrame.mnId) {
        mvpLocalKeyFrames.push_back(pParent);
        pParent->mnTrackReferenceForFrame = mCurrentFrame.mnId;
        break;
      }
    }
  }

  if (pKFmax) {
    mpReferenceKF = pKFmax;
    mCurrentFrame.mpReferenceKF = mpReferenceKF;
  }
}

bool Tracking::Relocalization() {
  // Compute Bag of Words Vector
  mCurrentFrame.ComputeBoW();

  // Relocalization is performed when tracking is lost
  // Track Lost: Query KeyFrame Database for keyframe candidates for
  // relocalisation
  vector<KeyFrame*> vpCandidateKFs;
  if (mpKeyFrameDB) {
    vpCandidateKFs =
        mpKeyFrameDB->DetectRelocalizationCandidates(&mCurrentFrame);
  }

  if (vpCandidateKFs.empty()) return false;

  const int nKFs = vpCandidateKFs.size();

  // We perform first an ORB matching with each candidate
  // If enough matches are found we setup a PnP solver
  ORBmatcher matcher(0.75, true);

  vector<PnPsolver*> vpPnPsolvers;
  vpPnPsolvers.resize(nKFs);

  vector<vector<MapPoint*>> vvpMapPointMatches;
  vvpMapPointMatches.resize(nKFs);

  vector<bool> vbDiscarded;
  vbDiscarded.resize(nKFs);

  int nCandidates = 0;

  for (int i = 0; i < nKFs; i++) {
    KeyFrame* pKF = vpCandidateKFs[i];
    if (pKF->isBad())
      vbDiscarded[i] = true;
    else {
      int nmatches =
          matcher.SearchByBoW(pKF, mCurrentFrame, vvpMapPointMatches[i]);
      if (nmatches < 15) {
        vbDiscarded[i] = true;
        continue;
      } else {
        PnPsolver* pSolver =
            new PnPsolver(mCurrentFrame, vvpMapPointMatches[i]);
        pSolver->SetRansacParameters(0.99, 10, 300, 4, 0.5, 5.991);
        vpPnPsolvers[i] = pSolver;
        nCandidates++;
      }
    }
  }

  // Alternatively perform some iterations of P4P RANSAC
  // Until we found a camera pose supported by enough inliers
  bool bMatch = false;
  ORBmatcher matcher2(0.9, true);

  while (nCandidates > 0 && !bMatch) {
    for (int i = 0; i < nKFs; i++) {
      if (vbDiscarded[i]) continue;

      // Perform 5 Ransac Iterations
      vector<bool> vbInliers;
      int nInliers;
      bool bNoMore;

      PnPsolver* pSolver = vpPnPsolvers[i];
      cv::Mat Tcw = pSolver->iterate(5, bNoMore, vbInliers, nInliers);

      // If Ransac reachs max. iterations discard keyframe
      if (bNoMore) {
        vbDiscarded[i] = true;
        nCandidates--;
      }

      // If a Camera Pose is computed, optimize
      if (!Tcw.empty()) {
        Tcw.copyTo(mCurrentFrame.mTcw);

        set<MapPoint*> sFound;

        const int np = vbInliers.size();

        for (int j = 0; j < np; j++) {
          if (vbInliers[j]) {
            mCurrentFrame.mvpMapPoints[j] = vvpMapPointMatches[i][j];
            sFound.insert(vvpMapPointMatches[i][j]);
          } else
            mCurrentFrame.mvpMapPoints[j] = NULL;
        }

        int nGood =
            Optimizer::PoseOptimization(&mCurrentFrame, mbUnsupervisedLearning);

        if (nGood < 10) continue;

        for (int io = 0; io < mCurrentFrame.N; io++)
          if (mCurrentFrame.mvbOutlier[io])
            mCurrentFrame.mvpMapPoints[io] = static_cast<MapPoint*>(NULL);

        // If few inliers, search by projection in a coarse window and optimize
        // again
        if (nGood < 50) {
          int nadditional = matcher2.SearchByProjection(
              mCurrentFrame, vpCandidateKFs[i], sFound, 10, 100);

          if (nadditional + nGood >= 50) {
            nGood = Optimizer::PoseOptimization(&mCurrentFrame,
                                                mbUnsupervisedLearning);

            // If many inliers but still not enough, search by projection again
            // in a narrower window the camera has been already optimized with
            // many points
            if (nGood > 30 && nGood < 50) {
              sFound.clear();
              for (int ip = 0; ip < mCurrentFrame.N; ip++)
                if (mCurrentFrame.mvpMapPoints[ip])
                  sFound.insert(mCurrentFrame.mvpMapPoints[ip]);
              nadditional = matcher2.SearchByProjection(
                  mCurrentFrame, vpCandidateKFs[i], sFound, 3, 64);

              // Final optimization
              if (nGood + nadditional >= 50) {
                nGood = Optimizer::PoseOptimization(&mCurrentFrame,
                                                    mbUnsupervisedLearning);

                for (int io = 0; io < mCurrentFrame.N; io++)
                  if (mCurrentFrame.mvbOutlier[io])
                    mCurrentFrame.mvpMapPoints[io] = NULL;
              }
            }
          }
        }

        // If the pose is supported by enough inliers stop ransacs and continue
        if (nGood >= 50) {
          bMatch = true;
          break;
        }
      }
    }
  }

  if (!bMatch) {
    return false;
  } else {
    mnLastRelocFrameId = mCurrentFrame.mnId;
    return true;
  }
}

void Tracking::SaveIntrospectionDataset() {
  if (mDatasetCreator) {
    mDatasetCreator->SaveToFile();
  }

  if (mDatasetCreatorFull) {
    mDatasetCreatorFull->SaveToFile();
  }
}

void Tracking::SaveTrackingResults(bool saving_on_failure) {
  // Save tracked trajectory and failure time stamp to file
  bool save_in_kitti_format = false;
  if (mSensor != System::eSensor::MONOCULAR) {
    // Save in KITTI format as well as TUM if we are not in monocular mode.
    save_in_kitti_format = true;
  }

  int sub_session_id;
  if (saving_on_failure) {
    sub_session_id = mFailureCount - 1;
  } else {
    sub_session_id = mFailureCount;
  }
  stringstream ss_suffix;
  ss_suffix << setfill('0') << setw(3) << sub_session_id;
  string suffix = ss_suffix.str();
  string path_to_traj, path_to_traj_kitti, path_to_ts_kitti;
  string path_to_failure_log;
  if (mbSaveVisualizationsToFile) {
    string dir = mvSaveVisualizationPath + "/trajectory/";
    path_to_traj = dir + "KeyFrameTrajectory_TUM_" + suffix + ".txt";
    path_to_failure_log = mvSaveVisualizationPath + "/failure_log.txt";

    string dir_kitti = mvSaveVisualizationPath + "/trajectory_kitti/";
    path_to_traj_kitti = dir_kitti + "Trajectory_KITTI_" + suffix + ".txt";
    path_to_ts_kitti = dir_kitti + "KITTI_time_" + suffix + ".txt";

    CreateDirectory(mvSaveVisualizationPath);
    if (mFailureCount <= 0 || (mFailureCount <= 1 && saving_on_failure)) {
      RemoveDirectory(dir);
      RemoveDirectory(dir_kitti);
    }
    CreateDirectory(dir);
    if (save_in_kitti_format) {
      CreateDirectory(dir_kitti);
    }
  } else {
    path_to_traj = "KeyFrameTrajectory_TUM_" + suffix + ".txt";
    path_to_traj_kitti = "Trajectory_KITTI_" + suffix + ".txt";
    path_to_ts_kitti = "KITTI_time_" + suffix + ".txt";
    path_to_failure_log = "failure_log.txt";
  }

  mpSystem->SaveKeyFrameTrajectoryTUM(path_to_traj);
  if (save_in_kitti_format) {
    mpSystem->SaveTrajectoryKITTI(path_to_traj_kitti, path_to_ts_kitti);
  }

  // Saves time stamp of most recent failure to file
  std::ofstream failure_log;

  if (saving_on_failure) {
    // Overwrite the file if it is the first failure instance
    if (mFailureCount <= 1) {
      failure_log.open(path_to_failure_log.c_str(),
                       std::ofstream::out | std::ofstream::trunc);
    } else {
      failure_log.open(path_to_failure_log.c_str(),
                       std::ofstream::out | std::ofstream::app);
    }

    if (!failure_log) {
      LOG(FATAL) << "Cannot open file path_to_failure_log";
    }
    failure_log << std::setprecision(20) << mlFrameTimes.back() << std::endl;

    failure_log.close();
  } else if (mFailureCount == 0) {
    // If not saving on failure, we still want to make sure to delete old
    // records. If mFailureCount==0 and we are not saving on failure, it means
    // that there has been no failures in this session, so we wipe any
    // old failure log files that might be on the path
    failure_log.open(path_to_failure_log.c_str(),
                     std::ofstream::out | std::ofstream::trunc);
    failure_log.close();
  }
}

void Tracking::Reset() {
  cout << "System Reseting" << endl;
  if (mpViewer) {
    mpViewer->RequestStop();
    while (!mpViewer->isStopped()) usleep(3000);
  }

  // Reset Local Mapping
  cout << "Reseting Local Mapper...";
  if (mbSingleThreaded) {
    mpLocalMapper->ForceReset();
  } else {
    mpLocalMapper->RequestReset();
  }
  cout << " done" << endl;

  // Reset Loop Closing
  cout << "Reseting Loop Closing...";
  if (mpLoopClosing) {
    mpLoopClosing->RequestReset();
  }
  cout << " done" << endl;

  // Clear BoW Database
  cout << "Reseting Database...";
  if (mpKeyFrameDB) {
    mpKeyFrameDB->clear();
  }
  cout << " done" << endl;

  // Clear Map (this erase MapPoints and KeyFrames)
  mpMap->clear();

  //     KeyFrame::nNextId = 0;
  //     Frame::nNextId = 0;
  mState = NO_IMAGES_YET;

  if (mpInitializer) {
    delete mpInitializer;
    mpInitializer = static_cast<Initializer*>(NULL);
  }

  mlRelativeFramePoses.clear();
  mlpReferences.clear();
  mlFrameTimes.clear();
  mlbLost.clear();

  if (mpViewer) mpViewer->Release();
}

void Tracking::ChangeCalibration(const string& strSettingPath) {
  cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);
  float fx = fSettings["Camera.fx"];
  float fy = fSettings["Camera.fy"];
  float cx = fSettings["Camera.cx"];
  float cy = fSettings["Camera.cy"];

  cv::Mat K = cv::Mat::eye(3, 3, CV_32F);
  K.at<float>(0, 0) = fx;
  K.at<float>(1, 1) = fy;
  K.at<float>(0, 2) = cx;
  K.at<float>(1, 2) = cy;
  K.copyTo(mK);

  cv::Mat DistCoef(4, 1, CV_32F);
  DistCoef.at<float>(0) = fSettings["Camera.k1"];
  DistCoef.at<float>(1) = fSettings["Camera.k2"];
  DistCoef.at<float>(2) = fSettings["Camera.p1"];
  DistCoef.at<float>(3) = fSettings["Camera.p2"];
  const float k3 = fSettings["Camera.k3"];
  if (k3 != 0) {
    DistCoef.resize(5);
    DistCoef.at<float>(4) = k3;
  }
  DistCoef.copyTo(mDistCoef);

  mbf = fSettings["Camera.bf"];

  Frame::mbInitialComputations = true;
}

void Tracking::InformOnlyTracking(const bool& flag) { mbOnlyTracking = flag; }

void Tracking::SetRelativeCamPoseUncertainty(
    const std::unordered_map<std::string, int>* pose_unc_map,
    const vector<Eigen::Vector2f>* rel_cam_poses_uncertainty) {
  mbRelCamPoseUncertaintyAvailable = true;
  mpPoseUncMap = pose_unc_map;
  mvpRelPoseUnc = rel_cam_poses_uncertainty;

  mFeatureEvaluator->SetRelativeCamPoseUncertainty(pose_unc_map,
                                                   rel_cam_poses_uncertainty);
}

cv::Mat Tracking::CalculateInverseTransform(const cv::Mat& transform) {
  if (transform.empty()) {
    std::cout << "MATRIX IS EMPTY!! " << std::endl;
  }

  cv::Mat R1 = transform.rowRange(0, 3).colRange(0, 3);
  cv::Mat t1 = transform.rowRange(0, 3).col(3);
  cv::Mat R1_inv = R1.t();
  cv::Mat t1_inv = -R1_inv * t1;
  cv::Mat transform_inv = cv::Mat::eye(4, 4, transform.type());

  R1_inv.copyTo(transform_inv.rowRange(0, 3).colRange(0, 3));
  t1_inv.copyTo(transform_inv.rowRange(0, 3).col(3));

  return transform_inv;
}

void Tracking::Release() {
  delete mpORBextractorLeft;
  delete mFeatureEvaluator;

  if (mSensor == System::STEREO) {
    delete mpORBextractorRight;
  }

  if (mSensor == System::MONOCULAR) {
    delete mpIniORBextractor;
  }

  if (mDatasetCreator) {
    delete mDatasetCreator;
  }
}

}  // namespace ORB_SLAM2
