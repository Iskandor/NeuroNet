cmake_minimum_required(VERSION 3.3)
project(libNeuroNet)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES
    include/BackProp.h
    include/BanditGame.h
    include/Connection.h
    include/Define.h
    include/IEnvironment.h
    include/matrix2.h
    include/matrix3.h
    include/Maze.h
    include/NBandit.h
    include/NeuralGroup.h
    include/NeuralNetwork.h
    include/sample_bp.h
    include/sample_td.h
    include/TD.h
    include/TDLambda.h
    include/vectorN.h
    src/BackProp.cpp
    src/BanditGame.cpp
    src/Connection.cpp
    src/Maze.cpp
    src/NBandit.cpp
    src/NeuralGroup.cpp
    src/NeuralNetwork.cpp
    src/sample_bp.cpp
    src/sample_td.cpp
    src/TDLambda.cpp
    main.cpp)

add_executable(libNeuroNet ${SOURCE_FILES})