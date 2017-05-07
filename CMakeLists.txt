cmake_minimum_required(VERSION 3.3)
project(NeuroNet)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -static-libstdc++ -static -lpthread") #-static-libstdc++

set(SOURCE_FILES
        algorithm/optimizer/BackProp.h
        network/Connection.h
        network/Define.h
        network/NeuralGroup.h
        network/NeuralNetwork.h
        algorithm/optimizer/BackProp.cpp
        network/Connection.cpp
        network/NeuralGroup.cpp
        network/NeuralNetwork.cpp
        samples/sample_bp.cpp
        log/log.h
        main.cpp
        algorithm/GradientDescent.cpp
        algorithm/GradientDescent.h
        algorithm/rl/QLearning.cpp
        algorithm/rl/QLearning.h
        samples/sampleMazeRL.cpp
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
        backend/flab/RandomGenerator.cpp
        backend/flab/RandomGenerator.h
        dataset/SequenceDataset.cpp
        dataset/SequenceDataset.h
        samples/sampleMSOM.cpp
        network/som/RecSOM.cpp
        network/som/RecSOM.h
        network/json.hpp
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
        algorithm/optimizer/RMSProp.cpp
        algorithm/optimizer/RMSProp.h
        algorithm/optimizer/Optimizer.cpp
        algorithm/optimizer/Optimizer.h
        backend/flab/Matrix.cpp
        backend/flab/Matrix.h
        algorithm/optimizer/ADAM.cpp
        algorithm/optimizer/ADAM.h
        algorithm/rl/SARSA.cpp
        algorithm/rl/SARSA.h
        algorithm/rl/ActorLearning.cpp
        algorithm/rl/ActorLearning.h
        backend/flab/Vector.cpp
        backend/flab/Vector.h
        backend/flab/Base.cpp
        backend/flab/Base.h
        samples/sampleALE.cpp
        environments/cartpole/CartPole.cpp
        environments/cartpole/CartPole.h
        samples/CartPoleTask.cpp
        samples/CartPoleTask.h
        samples/sampleCartPoleRL.cpp
        environments/cartpole/rk4.cpp
        environments/cartpole/rk4.hpp
        algorithm/rl/TD.cpp
        algorithm/rl/TD.h
        algorithm/rl/CACLAActor.cpp
        algorithm/rl/CACLAActor.h
        samples/sampleMazeRL.h
        samples/sampleCartPoleRL.h
        algorithm/optimizer/TDBP.cpp
        algorithm/optimizer/TDBP.h
        backend/flab/Tensor3.cpp
        backend/flab/Tensor3.h)

include_directories("C:/Users/mpechac/Documents/Arcade-Learning-Environment-0.5.1/src")

add_executable(NeuroNet-bin ${SOURCE_FILES})
add_library(NeuroNet STATIC ${SOURCE_FILES} NeuroNet.h algorithm/Algorithm.h dataset/DatasetCore.h network/Network.h)
set_target_properties(NeuroNet-bin PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)
set_target_properties(NeuroNet PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)
#target_link_libraries(NeuroNet-bin C:/Users/mpechac/Documents/NeuroNet/lib/libale.dll.a)
