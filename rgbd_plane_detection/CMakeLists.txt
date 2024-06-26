cmake_minimum_required (VERSION 3.1)

project (rgbd_plane_detection)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "-fpermissive")
# set(CMAKE_CXX_COMPILER /usr/bin/g++)
# set(CMAKE_BUILD_TYPE "Release")

find_package(catkin REQUIRED COMPONENTS
  roscpp
  sensor_msgs
  std_msgs
  cv_bridge
  image_transport
  message_filters
  message_generation
)


# put FindGLFW.cmake in the source directory of CMakeLists.txt
find_package(OpenCV 4 REQUIRED)
include_directories(
 include
  ${catkin_INCLUDE_DIRS}
)



# add_service_files(
#   FILES 
#   plane_detector.srv
# )

# generate_messages(
#   DEPENDENCIES
#   std_msgs
# )

catkin_package(
  CATKIN_DEPENDS roscpp std_msgs sensor_msgs message_runtime
)
include_directories(${OpenCV_INCLUDE_DIRS} /usr/include/eigen3)
include_directories(include/peac)
include_directories(include/MRF2.2)

link_directories(include/MRF2.2)

file(GLOB sources "src/*.cpp")

add_executable(rgbd_plane_detection ${sources})

target_link_libraries(rgbd_plane_detection 
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS} libMRF.a)