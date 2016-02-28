cmake_minimum_required(VERSION 3.3)
project(libNeuroNet)

#                                               -*- cmake -*-
#
#  Eigen3Config.cmake(.in)

# Use the following variables to compile and link against Eigen:
#  EIGEN3_FOUND              - True if Eigen was found on your system
#  EIGEN3_USE_FILE           - The file making Eigen usable
#  EIGEN3_DEFINITIONS        - Definitions needed to build with Eigen
#  EIGEN3_INCLUDE_DIR        - Directory where signature_of_eigen3_matrix_library can be found
#  EIGEN3_INCLUDE_DIRS       - List of directories of Eigen and it's dependencies
#  EIGEN3_ROOT_DIR           - The base directory of Eigen
#  EIGEN3_VERSION_STRING     - A human-readable string containing the version
#  EIGEN3_VERSION_MAJOR      - The major version of Eigen
#  EIGEN3_VERSION_MINOR      - The minor version of Eigen
#  EIGEN3_VERSION_PATCH      - The patch version of Eigen

set ( EIGEN3_FOUND 1 )
set ( EIGEN3_USE_FILE     "${CMAKE_CURRENT_LIST_DIR}/UseEigen3.cmake" )

set ( EIGEN3_DEFINITIONS  "" )
set ( EIGEN3_INCLUDE_DIR  "C:/GIT/lib/eigen/lib/include/eigen3" )
set ( EIGEN3_INCLUDE_DIRS "C:/GIT/libNeuroNet/lib/eigen3/include/eigen3" )
set ( EIGEN3_ROOT_DIR     "C:/GIT/libNeuroNet/lib/eigen3" )



set ( EIGEN3_VERSION_STRING "3.2.92" )
set ( EIGEN3_VERSION_MAJOR  "3" )
set ( EIGEN3_VERSION_MINOR  "2" )
set ( EIGEN3_VERSION_PATCH  "92" )

add_definitions     ( ${EIGEN3_DEFINITIONS} )
include_directories ( ${EIGEN3_INCLUDE_DIRS} )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES
        algorithm/BackProp.h
        samples/BanditGame.h
        network/Connection.h
        network/Define.h
        algorithm/IEnvironment.h
        samples/Maze.h
        samples/NBandit.h
        network/NeuralGroup.h
        network/NeuralNetwork.h
        samples/sample_bp.h
        algorithm/BackProp.cpp
        samples/BanditGame.cpp
        network/Connection.cpp
        samples/Maze.cpp
        samples/NBandit.cpp
        network/NeuralGroup.cpp
        network/NeuralNetwork.cpp
        samples/sample_bp.cpp
        algorithm/TDLambda.cpp
        algorithm/TDLambda.h
        samples/sample_td.cpp
        samples/sample_td.h
    main.cpp algorithm/GradientBase.cpp algorithm/GradientBase.h algorithm/QLearning.cpp algorithm/QLearning.h samples/sampleQ.cpp samples/sampleQ.h samples/LunarLander.cpp samples/LunarLander.h)

add_executable(libNeuroNet ${SOURCE_FILES})