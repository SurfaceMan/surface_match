cmake_minimum_required(VERSION 3.10)
project(ppf)

set(CMAKE_CXX_STANDARD 17)

include(FetchContent)
FetchContent_Declare(
    gtl
    GIT_REPOSITORY https://github.com/greg7mdp/gtl.git
    GIT_TAG v1.1.5 # adjust tag/branch/commit as needed
)
FetchContent_MakeAvailable(gtl)

# find OpenMP
find_package(OpenMP REQUIRED)

if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

set(EIGEN3_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen)
set(XSIMD_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/xsimd/include)
set(KDTREE_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/kdtree)
set(RPLY_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/rply/rply)
set(RPLY_SOURCE ${CMAKE_CURRENT_SOURCE_DIR}/third_party/rply/rply/rply.c)

add_library(ppf SHARED
    ppf.h ppf.cpp util.h util.cpp type.h type.cpp icp.h icp.cpp privateUtil.h
    filePLY.h filePLY.cpp ${RPLY_SOURCE} serialize.h serialize.cpp privateType.h
)

target_include_directories(ppf PRIVATE
    ${KDTREE_INCLUDE_DIRS}
    ${XSIMD_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${RPLY_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_libraries(ppf PUBLIC OpenMP::OpenMP_CXX gtl)
target_compile_options(ppf PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/bigobj /arch:AVX2>
    $<$<CXX_COMPILER_ID:GNU>: -mavx -fPIC -fvisibility=hidden -Wl,--exclude-libs,ALL>
)
target_compile_definitions(ppf PUBLIC API_EXPORTS)

add_library(ppf::ppf ALIAS ppf)
set_property(TARGET ppf PROPERTY EXPORT_NAME ppf::ppf)

install(FILES ppf.h type.h apiExport.h filePLY.h util.h TYPE INCLUDE)
install(TARGETS ppf EXPORT ppf LIBRARY)

# install(EXPORT ppf FILE ppfTargets.cmake DESTINATION lib/cmake/)
# install(EXPORT ppf FILE ppf-config.cmake DESTINATION lib/cmake/)
option(BUILD_EXAMPLE "build example" ON)

if(BUILD_EXAMPLE)
    add_executable(test_ppf main.cpp)
    target_include_directories(test_ppf PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${EIGEN3_INCLUDE_DIRS})
    target_link_libraries(test_ppf ppf::ppf)
endif(BUILD_EXAMPLE)