cmake_minimum_required(VERSION 3.10)
project(ppf)

find_package(Eigen3)

#find OpenMP 
find_package(OpenMP REQUIRED)
if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

add_executable(ppf  
    main.cpp ppf.h ppf.cpp util.h util.cpp type.h type.cpp icp.h icp.cpp KDTreeVectorOfVectorsAdaptor.h
)

target_include_directories(ppf PRIVATE 
    "D:/tools/PCL 1.12.1/3rdParty/Eigen/eigen3"
    "E:/project2/2022-2-9ppf/nanoflann/include"
    "/home/abc/project/3_surface_match/nanoflann/include"
    ${EIGEN3_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(ppf PUBLIC OpenMP::OpenMP_CXX)
