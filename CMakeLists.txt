cmake_minimum_required(VERSION 3.16)
project(VGGTSyncMultiCamDemo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(OpenCV_DIR "D:/code/opencv4.11.0/build/")
find_package(OpenCV REQUIRED core imgcodecs imgproc videoio)
add_executable(minimal_video_read_test src/minimal_video_read_test.cpp)
target_link_libraries(minimal_video_read_test PRIVATE ${OpenCV_LIBS})
#将dll复制到可执行文件所在目录
add_custom_command(TARGET minimal_video_read_test POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
            "D:/code/opencv4.11.0/build/x64/vc16/bin"
            $<TARGET_FILE_DIR:minimal_video_read_test>)

