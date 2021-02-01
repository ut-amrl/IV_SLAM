echo "Building ROS nodes"

cd Examples/ROS/ORB_SLAM2
export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:`pwd`
mkdir build
cd build
cmake .. -DROS_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=../../../Thirdparty/libtorch
make -j
