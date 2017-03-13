cmake_minimum_required(VERSION 3.3)
project(NeuroNet)

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
set ( EIGEN3_INCLUDE_DIR  "./lib/include/eigen3" )
set ( EIGEN3_INCLUDE_DIRS "./lib/eigen3/include/eigen3" )
set ( EIGEN3_ROOT_DIR     "./lib/eigen3" )



set ( EIGEN3_VERSION_STRING "3.2.92" )
set ( EIGEN3_VERSION_MAJOR  "3" )
set ( EIGEN3_VERSION_MINOR  "2" )
set ( EIGEN3_VERSION_PATCH  "92" )

add_definitions     ( ${EIGEN3_DEFINITIONS} )
include_directories ( ${EIGEN3_INCLUDE_DIRS} )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -static-libgcc -static-libstdc++ -static -lpthread")

set(SOURCE_FILES
        algorithm/BackProp.h
        network/Connection.h
        network/Define.h
        algorithm/IEnvironment.h
        samples/MazeOld.h
        network/NeuralGroup.h
        network/NeuralNetwork.h
        algorithm/BackProp.cpp
        network/Connection.cpp
        samples/MazeOld.cpp
        network/NeuralGroup.cpp
        network/NeuralNetwork.cpp
        samples/sample_bp.cpp
        log/log.h
        main.cpp
        algorithm/GradientDescent.cpp
        algorithm/GradientDescent.h
        algorithm/rl/QLearning.cpp
        algorithm/rl/QLearning.h
        samples/sampleQ.cpp
        samples/LunarLander.cpp
        samples/LunarLander.h
        network/som/SOM.cpp
        network/som/SOM.h
        network/som/MSOM.cpp
        network/som/MSOM.h
        dataset/Dataset.cpp
        dataset/Dataset.h
        dataset/StringUtils.cpp
        dataset/StringUtils.h
        samples/sampleSOM.cpp
        dataset/DatasetConfig.h
        algorithm/GreedyPolicy.cpp
        algorithm/GreedyPolicy.h
        samples/samples.h
        network/NetworkUtils.cpp
        network/NetworkUtils.h
        network/filters/IFilter.h
        network/filters/NormalizationFilter.cpp
        network/filters/NormalizationFilter.h
        network/filters/KwtaFilter.cpp
        network/filters/KwtaFilter.h
        algorithm/LearningAlgorithm.cpp
        algorithm/LearningAlgorithm.h
        network/RandomGenerator.cpp
        network/RandomGenerator.h
        dataset/SequenceDataset.cpp
        dataset/SequenceDataset.h
        samples/sampleMSOM.cpp
        network/som/RecSOM.cpp
        network/som/RecSOM.h
        network/json.hpp
		samples/PoleCart.cpp
		samples/PoleCart.h
		samples/MountainCar.cpp
		samples/MountainCar.h
        algorithm/rec/RTRL.cpp
        algorithm/rec/RTRL.h
        environments/Environment.cpp
        environments/Environment.h
        environments/maze/Maze.cpp
        environments/maze/Maze.h
        environments/maze/MazeAction.cpp
        environments/maze/MazeAction.h
        samples/MazeTask.cpp
        samples/MazeTask.h
        algorithm/RMSProp.cpp
        algorithm/RMSProp.h
        algorithm/Optimizer.cpp
        algorithm/Optimizer.h
        backend/Matrix.cpp
        backend/Matrix.h algorithm/ADAM.cpp algorithm/ADAM.h)

add_executable(NeuroNet-bin ${SOURCE_FILES})
add_library(NeuroNet STATIC ${SOURCE_FILES} NeuroNet.h algorithm/Algorithm.h dataset/DatasetCore.h network/Network.h)
set_target_properties(NeuroNet-bin PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)
set_target_properties(NeuroNet PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)
