
cmake_minimum_required(VERSION 2.8)


#设置cmake的最小版本 
cmake_minimum_required(VERSION 2.8) 
#项目名称 
project(demo) 
#设置c++编译器 
set(CMAKE_CXX_STANDARD 11)
#设置TENSORFLOW_DIR变量，变量内容为安装的tensorflow文件夹路径 
set(TENSORFLOW_DIR /home/user/cpp/tensorflow) 

find_package(OpenCV REQUIRED)
#项目中的include路径 
include_directories(${TENSORFLOW_DIR}) 
include_directories(${TENSORFLOW_DIR}/tensorflow/contrib/makefile/gen/proto)
include_directories(${TENSORFLOW_DIR}/bazel-genfiles)
include_directories(${TENSORFLOW_DIR}/bazel-bin/tensorflow )
include_directories(${TENSORFLOW_DIR}/tensorflow/contrib/makefile/gen/protobuf/include) 
include_directories(${TENSORFLOW_DIR}/tensorflow/contrib/makefile/downloads/eigen) 
include_directories(${TENSORFLOW_DIR}/tensorflow/contrib/makefile/downloads/nsync/public)
include_directories(${TENSORFLOW_DIR}/tensorflow/contrib/makefile/downloads/absl) 
#项目中lib路径 
link_directories(${TENSORFLOW_DIR}/tensorflow/contrib/makefile/gen/lib) 
link_directories(${TENSORFLOW_DIR}/tensorflow/contrib/makefile/gen/protobuf-host/lib) 
link_directories(${TENSORFLOW_DIR}/tensorflow/contrib/makefile/downloads/nsync/builds/default.linux.c++11) 
link_directories(${TENSORFLOW_DIR}/bazel-bin/tensorflow) 
add_executable(demo cv.cpp) 
#连接libtensorflow_cc.so和libtensorflow_framework库。 
target_link_libraries(demo tensorflow_cc tensorflow_framework ${OpenCV_LIBS})


#cmake_minimum_required(VERSION 2.8)
#project(opencv)
#find_package(OpenCV REQUIRED)
#add_executable(opencv try.cpp)
#target_link_libraries(opencv ${OpenCV_LIBS})
