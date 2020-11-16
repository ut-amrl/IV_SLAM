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

#include "MapDrawer.h"
#include "MapPoint.h"
#include "KeyFrame.h"
#include <pangolin/pangolin.h>
#include <mutex>
#include <glog/logging.h>

DEFINE_bool(map_drawer_visualize_gt_pose, true, 
            "Visualizes the reference camera poses if available. ");

namespace ORB_SLAM2
{


MapDrawer::MapDrawer(Map* pMap, const string &strSettingPath):mpMap(pMap)
{
    cv::FileStorage fSettings(strSettingPath, cv::FileStorage::READ);

    mKeyFrameSize = fSettings["Viewer.KeyFrameSize"];
    mKeyFrameLineWidth = fSettings["Viewer.KeyFrameLineWidth"];
    mGraphLineWidth = fSettings["Viewer.GraphLineWidth"];
    mPointSize = fSettings["Viewer.PointSize"];
    mCameraSize = fSettings["Viewer.CameraSize"];
    mCameraLineWidth = fSettings["Viewer.CameraLineWidth"];

}

void MapDrawer::DrawMapPoints()
{
    // Color points given their quality score
    const bool kColorPointsQual = true;
  
    const vector<MapPoint*> &vpMPs = mpMap->GetAllMapPoints();
    const vector<MapPoint*> &vpRefMPs = mpMap->GetReferenceMapPoints();

    set<MapPoint*> spRefMPs(vpRefMPs.begin(), vpRefMPs.end());

    if(vpMPs.empty())
        return;

    glPointSize(mPointSize);
    glBegin(GL_POINTS);
    glColor3f(0.0,0.0,0.0);

    for(size_t i=0, iend=vpMPs.size(); i<iend;i++)
    {
        if(vpMPs[i]->isBad() || spRefMPs.count(vpMPs[i]))
            continue;
        
        // Do not visualize map points with few observations
        if(false && vpMPs[i]->Observations() <0) {
          continue;
        }
        
        if(!vpMPs[i]->mbQualityScoreCalculated) {
          // Cyan: Map points that for which, quality score has not been
          // calculated. Note that "reference" map points will be recolored 
          // differently
          glColor3f(0.0,1.0,1.0);
        } else {
          glColor3f(1.0 - vpMPs[i]->GetQualityScore(), 
                          vpMPs[i]->GetQualityScore(),
                          0.0);
        }
        
        cv::Mat pos = vpMPs[i]->GetWorldPos();
        glVertex3f(pos.at<float>(0),pos.at<float>(1),pos.at<float>(2));
    }
    glEnd();

    glPointSize(mPointSize);
    glBegin(GL_POINTS);
    glColor3f(1.0,0.0,0.0);

    for(set<MapPoint*>::iterator sit=spRefMPs.begin(), send=spRefMPs.end(); 
sit!=send; sit++)
    {
        if((*sit)->isBad())
            continue;
        
        // Do not visualize map points with few observations
        if((*sit)->Observations() < 0) {
          continue;
        }
        
        if(kColorPointsQual) {
          if (false && (*sit)->Observations() < 2) {
            // Cyan: Very short track map point
            glColor3f(0.0,1.0,1.0);
          } else if ((*sit)->mbQualityScoreCalculated) {
            // Green at high quality, red at low quality
            glColor3f(1.0 - (*sit)->GetQualityScore(), 
                      (*sit)->GetQualityScore(),
                      0.0);
          } else {
            // Yellow: Quality score unknown
            glColor3f(1.0, 1.0, 0.0);
          }
        }
        
        cv::Mat pos = (*sit)->GetWorldPos();
        glVertex3f(pos.at<float>(0),pos.at<float>(1),pos.at<float>(2));
    }

    glEnd();
}

void MapDrawer::DrawKeyFrames(const bool bDrawKF, const bool bDrawGraph)
{
    const float &w = mKeyFrameSize;
    const float h = w*0.75;
    const float z = w*0.6;

    const vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();

    if(bDrawKF)
    {
        for(size_t i=0; i<vpKFs.size(); i++)
        {
            KeyFrame* pKF = vpKFs[i];
            cv::Mat Twc = pKF->GetPoseInverse().t();

            glPushMatrix();

            glMultMatrixf(Twc.ptr<GLfloat>(0));

            glLineWidth(mKeyFrameLineWidth);
            glColor3f(0.0f,0.0f,1.0f);
            glBegin(GL_LINES);
            glVertex3f(0,0,0);
            glVertex3f(w,h,z);
            glVertex3f(0,0,0);
            glVertex3f(w,-h,z);
            glVertex3f(0,0,0);
            glVertex3f(-w,-h,z);
            glVertex3f(0,0,0);
            glVertex3f(-w,h,z);

            glVertex3f(w,h,z);
            glVertex3f(w,-h,z);

            glVertex3f(-w,h,z);
            glVertex3f(-w,-h,z);

            glVertex3f(-w,h,z);
            glVertex3f(w,h,z);

            glVertex3f(-w,-h,z);
            glVertex3f(w,-h,z);
            glEnd();

            glPopMatrix();
        }
        
        // Draw the ground truth keyframe poses for the last 
        // mGTCameraPoseHistory keframes
        if (FLAGS_map_drawer_visualize_gt_pose && mbGTPoseAvailable) {
          size_t st_idx = std::max(0, static_cast<int>(vpKFs.size()) - 
                                      mGTCameraPoseHistory);
          for(size_t i=st_idx; i<vpKFs.size(); i++){
            KeyFrame* pKF = vpKFs[i];
            cv::Mat Twc_gt = pKF->GetGTPose();
            if (i == st_idx) {
              mTwc_gt_vis_init = Twc_gt;
              mTwc_vis_init = pKF->GetPoseInverse();
            }
            
            // Calculate the ground truth pose of current keyframe wrt 
            // mTwc_gt_vis_init and then overlay mTwc_gt_vis_init on 
            // mTwc_vis_init
            cv::Mat Twc_gt_transformed =  mTwc_vis_init * 
                        CalculateRelativeTransform(mTwc_gt_vis_init, Twc_gt);
            
            Twc_gt_transformed = Twc_gt_transformed.t();

            glPushMatrix();

            glMultMatrixf(Twc_gt_transformed.ptr<GLfloat>(0));

            glLineWidth(mKeyFrameLineWidth);
            glColor3f(1.0f,0.0f,0.0f);
            glBegin(GL_LINES);
            glVertex3f(0,0,0);
            glVertex3f(w,h,z);
            glVertex3f(0,0,0);
            glVertex3f(w,-h,z);
            glVertex3f(0,0,0);
            glVertex3f(-w,-h,z);
            glVertex3f(0,0,0);
            glVertex3f(-w,h,z);

            glVertex3f(w,h,z);
            glVertex3f(w,-h,z);

            glVertex3f(-w,h,z);
            glVertex3f(-w,-h,z);

            glVertex3f(-w,h,z);
            glVertex3f(w,h,z);

            glVertex3f(-w,-h,z);
            glVertex3f(w,-h,z);
            glEnd();

            glPopMatrix();
          }
        }
    }

    if(bDrawGraph)
    {
        glLineWidth(mGraphLineWidth);
//         glColor4f(0.0f,1.0f,0.0f,0.6f); // green
        glColor4f(0.0f,0.0f,1.0f,0.6f); // blue
        glBegin(GL_LINES);

        for(size_t i=0; i<vpKFs.size(); i++)
        {
            // Covisibility Graph
            const vector<KeyFrame*> vCovKFs = 
vpKFs[i]->GetCovisiblesByWeight(100);
            cv::Mat Ow = vpKFs[i]->GetCameraCenter();
            if(!vCovKFs.empty())
            {
                for(vector<KeyFrame*>::const_iterator vit=vCovKFs.begin(), 
vend=vCovKFs.end(); vit!=vend; vit++)
                {
                    if((*vit)->mnId<vpKFs[i]->mnId)
                        continue;
                    cv::Mat Ow2 = (*vit)->GetCameraCenter();
                    glVertex3f(Ow.at<float>(0),Ow.at<float>(1),Ow.at<float>(2));
                    
glVertex3f(Ow2.at<float>(0),Ow2.at<float>(1),Ow2.at<float>(2));
                }
            }

            // Spanning tree
            KeyFrame* pParent = vpKFs[i]->GetParent();
            if(pParent)
            {
                cv::Mat Owp = pParent->GetCameraCenter();
                glVertex3f(Ow.at<float>(0),Ow.at<float>(1),Ow.at<float>(2));
                glVertex3f(Owp.at<float>(0),Owp.at<float>(1),Owp.at<float>(2));
            }

            // Loops
            set<KeyFrame*> sLoopKFs = vpKFs[i]->GetLoopEdges();
            for(set<KeyFrame*>::iterator sit=sLoopKFs.begin(), 
send=sLoopKFs.end(); sit!=send; sit++)
            {
                if((*sit)->mnId<vpKFs[i]->mnId)
                    continue;
                cv::Mat Owl = (*sit)->GetCameraCenter();
                glVertex3f(Ow.at<float>(0),Ow.at<float>(1),Ow.at<float>(2));
                glVertex3f(Owl.at<float>(0),Owl.at<float>(1),Owl.at<float>(2));
            }
        }

        glEnd();
    }
}

void MapDrawer::DrawCurrentCamera(pangolin::OpenGlMatrix &Twc)
{
    const float &w = mCameraSize;
    const float h = w*0.75;
    const float z = w*0.6;

    glPushMatrix();

#ifdef HAVE_GLES
        glMultMatrixf(Twc.m);
#else
        glMultMatrixd(Twc.m);
#endif

    glLineWidth(mCameraLineWidth);
//     glColor3f(0.0f,1.0f,0.0f); // green
    glColor3f(0.0f,0.0f,1.0f); // blue
    glBegin(GL_LINES);
    glVertex3f(0,0,0);
    glVertex3f(w,h,z);
    glVertex3f(0,0,0);
    glVertex3f(w,-h,z);
    glVertex3f(0,0,0);
    glVertex3f(-w,-h,z);
    glVertex3f(0,0,0);
    glVertex3f(-w,h,z);

    glVertex3f(w,h,z);
    glVertex3f(w,-h,z);

    glVertex3f(-w,h,z);
    glVertex3f(-w,-h,z);

    glVertex3f(-w,h,z);
    glVertex3f(w,h,z);

    glVertex3f(-w,-h,z);
    glVertex3f(w,-h,z);
    glEnd();

    glPopMatrix();
}

void MapDrawer::DrawCurrentCameraGT() {
    if (mbGTPoseAvailable && FLAGS_map_drawer_visualize_gt_pose && 
!mTwc_gt_vis_init.empty()) {
      const float &w = mCameraSize;
      const float h = w*0.75;
      const float z = w*0.6;
    
      // Calculate the ground truth pose of current frame wrt 
      // mTwc_gt_vis_init and then overlay mTwc_gt_vis_init on 
      // mTwc_vis_init
      cv::Mat Twc_gt_transformed;
      
      {
        unique_lock<mutex> lock(mMutexCamera);
        Twc_gt_transformed =  mTwc_vis_init * 
                  CalculateRelativeTransform(mTwc_gt_vis_init, mTwc_gt);
      }
    
      // Convert mTwc_gt to OpenGlMatrix
      pangolin::OpenGlMatrix M;
      ConverT2OpenGLCameraMatrix(Twc_gt_transformed, M);

      glPushMatrix();

  #ifdef HAVE_GLES
          glMultMatrixf(M.m);
  #else
          glMultMatrixd(M.m);
  #endif

      glLineWidth(mCameraLineWidth);
      glColor3f(1.0f,0.5f,0.0f);
      glBegin(GL_LINES);
      glVertex3f(0,0,0);
      glVertex3f(w,h,z);
      glVertex3f(0,0,0);
      glVertex3f(w,-h,z);
      glVertex3f(0,0,0);
      glVertex3f(-w,-h,z);
      glVertex3f(0,0,0);
      glVertex3f(-w,h,z);

      glVertex3f(w,h,z);
      glVertex3f(w,-h,z);

      glVertex3f(-w,h,z);
      glVertex3f(-w,-h,z);

      glVertex3f(-w,h,z);
      glVertex3f(w,h,z);

      glVertex3f(-w,-h,z);
      glVertex3f(w,-h,z);
      glEnd();

      glPopMatrix();
    }
}


void MapDrawer::SetCurrentCameraPose(const cv::Mat &Tcw)
{
    unique_lock<mutex> lock(mMutexCamera);
    mCameraPose = Tcw.clone();
}

void MapDrawer::SetCurrentCameraPose(const cv::Mat &Tcw, 
                              const std::string& strFrameName) {
    unique_lock<mutex> lock(mMutexCamera);
    mCameraPose = Tcw.clone();
    mstrFrameName = strFrameName.substr(0, strFrameName.length()-4);
}

void MapDrawer::SetCurrentCameraPosewithGT(const cv::Mat &Tcw,
                                const cv::Mat &Twc_gt, 
                              const std::string& strFrameName){
    unique_lock<mutex> lock(mMutexCamera);
    mCameraPose = Tcw.clone();
    mTwc_gt = Twc_gt.clone();
    mbGTPoseAvailable = true;
    mstrFrameName = strFrameName.substr(0, strFrameName.length()-4);
}

void MapDrawer::GetCurrentOpenGLCameraMatrix(pangolin::OpenGlMatrix &M)
{
    if(!mCameraPose.empty())
    {
        cv::Mat Rwc(3,3,CV_32F);
        cv::Mat twc(3,1,CV_32F);
        {
            unique_lock<mutex> lock(mMutexCamera);
            Rwc = mCameraPose.rowRange(0,3).colRange(0,3).t();
            twc = -Rwc*mCameraPose.rowRange(0,3).col(3);
        }

        M.m[0] = Rwc.at<float>(0,0);
        M.m[1] = Rwc.at<float>(1,0);
        M.m[2] = Rwc.at<float>(2,0);
        M.m[3]  = 0.0;

        M.m[4] = Rwc.at<float>(0,1);
        M.m[5] = Rwc.at<float>(1,1);
        M.m[6] = Rwc.at<float>(2,1);
        M.m[7]  = 0.0;

        M.m[8] = Rwc.at<float>(0,2);
        M.m[9] = Rwc.at<float>(1,2);
        M.m[10] = Rwc.at<float>(2,2);
        M.m[11]  = 0.0;

        M.m[12] = twc.at<float>(0);
        M.m[13] = twc.at<float>(1);
        M.m[14] = twc.at<float>(2);
        M.m[15]  = 1.0;
    }
    else
        M.SetIdentity();
}

void MapDrawer::ConverT2OpenGLCameraMatrix(const cv::Mat &Twc,
                                    pangolin::OpenGlMatrix &M) {
    if(!Twc.empty())
    {
        M.m[0] = Twc.at<float>(0,0);
        M.m[1] = Twc.at<float>(1,0);
        M.m[2] = Twc.at<float>(2,0);
        M.m[3]  = 0.0;

        M.m[4] = Twc.at<float>(0,1);
        M.m[5] = Twc.at<float>(1,1);
        M.m[6] = Twc.at<float>(2,1);
        M.m[7]  = 0.0;

        M.m[8] = Twc.at<float>(0,2);
        M.m[9] = Twc.at<float>(1,2);
        M.m[10] = Twc.at<float>(2,2);
        M.m[11]  = 0.0;

        M.m[12] = Twc.at<float>(0,3);
        M.m[13] = Twc.at<float>(1,3);
        M.m[14] = Twc.at<float>(2,3);
        M.m[15]  = 1.0;
    }
    else
        M.SetIdentity();
}

cv::Mat MapDrawer::CalculateRelativeTransform(
        const cv::Mat& dest_frame_pose, 
        const cv::Mat& src_frame_pose) {  
  return CalculateInverseTransform(dest_frame_pose) * src_frame_pose;
}

cv::Mat MapDrawer::CalculateInverseTransform(const cv::Mat& transform) {
  if (transform.empty()) {
    std::cout << "MATRIX IS EMPTY!! " << std::endl;
  }
  
  cv::Mat R1 = transform.rowRange(0,3).colRange(0,3);
  cv::Mat t1 = transform.rowRange(0,3).col(3);
  cv::Mat R1_inv = R1.t();
  cv::Mat t1_inv = -R1_inv*t1;
  cv::Mat transform_inv = cv::Mat::eye(4,4, transform.type());
 
  R1_inv.copyTo(transform_inv.rowRange(0,3).colRange(0,3));
  t1_inv.copyTo(transform_inv.rowRange(0,3).col(3));
  
  return transform_inv;
}

} //namespace ORB_SLAM
