cmake_minimum_required(VERSION 3.10)
project(ppf)

find_package(Eigen3 REQUIRED)

add_executable(ppf  
    main.cpp ppf.h ppf.cpp util.h util.cpp type.h
)

target_include_directories(ppf PRIVATE 
    ${EIGEN3_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/
)
