#========================================
# ExemplarSVM-LIBSVM (ESVM)
#========================================

cmake_minimum_required(VERSION 2.8)

# project
set(ESVM_PROJECT "ESVM")
project(${ESVM_PROJECT})

if(MSVC)
    set(CMAKE_USE_RELATIVE_PATHS ON CACHE INTERNAL "" FORCE)
    add_definitions("/EHsc")
else()
    set(CMAKE_CXX_COMPILER g++)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC")
endif()

# build type
if(NOT CMAKE_DEBUG_POSTFIX)
  set(CMAKE_DEBUG_POSTFIX d)
endif()

# Offer the user the choice of overriding the installation directories
set(INSTALL_LIBRARY_DIR lib CACHE PATH "Installation directory for libraries")
set(INSTALL_BINARY_DIR  bin CACHE PATH "Installation directory for executables")
set(INSTALL_INCLUDE_DIR inc CACHE PATH "Installation directory for header files")
if(WIN32 AND NOT CYGWIN)
  set(DEF_INSTALL_CMAKE_DIR cmake)
else()
  set(DEF_INSTALL_CMAKE_DIR lib/CMake/${ESVM_PROJECT})
endif()
set(INSTALL_CMAKE_DIR ${DEF_INSTALL_CMAKE_DIR} CACHE PATH "Installation directory for CMake files")

# set ESVM paths
set(ESVM_ROOT_DIR           "${CMAKE_CURRENT_LIST_DIR}")
set(ESVM_EXECUTABLE_DIR     "${ESVM_ROOT_DIR}/bld")
set(ESVM_SOURCES_DIRS       "${ESVM_ROOT_DIR}/src")
set(ESVM_INCLUDE_DIRS       "${ESVM_ROOT_DIR}/inc")
set(ESVM_LIBRARY_DIR        "${ESVM_ROOT_DIR}/lib")
set(CMAKE_INSTALL_PREFIX    "${ESVM_ROOT_DIR}/install" CACHE STRING "")

# Make relative paths absolute (needed later on)
foreach(p LIBRARY BINARY INCLUDE CMAKE)
  set(var INSTALL_${p}_DIR)
  if(NOT IS_ABSOLUTE "${${var}}")
    set(${var} "${CMAKE_INSTALL_PREFIX}/${${var}}")
  endif()
endforeach()

# include cmake subdirectory to find packages
list(APPEND CMAKE_MODULE_PATH "${ESVM_ROOT_DIR}/cmake")

# ESVM options
option(ESVM_USE_LBP         "Employ FeatureExtractorLBP (feLBP) for ESVM"   OFF)
option(ESVM_BUILD_HOG       "Build FeatureExtractorHOG (feHOG) from source" OFF)
option(ESVM_BUILD_TESTS     "Build executable for tests"                    OFF)
if (MSVC)
    # under Windows, using LIBSVM library causes malloc/free errors
    # but building directly from the header/source is working
    option(ESVM_BUILD_LIBSVM    "Build LIBSVM from sources"                 OFF)
else()
    option(ESVM_BUILD_LIBSVM    "Build LIBSVM from sources"                 ON)
endif()

# drop down list of library type
set(ESVM_LIBRARY_STATIC    "STATIC")
set(ESVM_LIBRARY_SHARED    "SHARED")
set(ESVM_LIBRARY_TYPE ${ESVM_LIBRARY_STATIC} CACHE STRING "Library type to generate")
set_property(CACHE ESVM_LIBRARY_TYPE PROPERTY STRINGS ${ESVM_LIBRARY_STATIC} ${ESVM_LIBRARY_SHARED})
# drop down list of building test mode
if (${ESVM_BUILD_TESTS})
    set(ESVM_BUILD_LIBRARY "Build from ESVM library")
    set(ESVM_BUILD_SOURCES "Build from ESVM sources")
    set(ESVM_BUILD_TESTS_MODE ${ESVM_BUILD_LIBRARY} CACHE STRING "Build executable mode for tests")
    set_property(CACHE ESVM_BUILD_TESTS_MODE PROPERTY STRINGS ${ESVM_BUILD_LIBRARY} ${ESVM_BUILD_SOURCES})
else()
    unset(ESVM_BUILD_TESTS_MODE CACHE)
endif()

# find ESVM header/source files
set(ESVM_HEADER_FILES ${ESVM_HEADER_FILES} ${ESVM_INCLUDE_DIRS}/esvm.h)
set(ESVM_HEADER_FILES ${ESVM_HEADER_FILES} ${ESVM_INCLUDE_DIRS}/esvmEnsemble.h)
set(ESVM_HEADER_FILES ${ESVM_HEADER_FILES} ${ESVM_INCLUDE_DIRS}/esvmOptions.h)
set(ESVM_HEADER_FILES ${ESVM_HEADER_FILES} ${ESVM_INCLUDE_DIRS}/esvmTypes.h)
set(ESVM_HEADER_FILES ${ESVM_HEADER_FILES} ${ESVM_INCLUDE_DIRS}/esvmUtils.h)
set(ESVM_SOURCE_FILES ${ESVM_SOURCE_FILES} ${ESVM_SOURCES_DIRS}/esvm.cpp)
set(ESVM_SOURCE_FILES ${ESVM_SOURCE_FILES} ${ESVM_SOURCES_DIRS}/esvmEnsemble.cpp)
set(ESVM_SOURCE_FILES ${ESVM_SOURCE_FILES} ${ESVM_SOURCES_DIRS}/esvmUtils.cpp)
if (${ESVM_BUILD_TESTS})
    set(ESVM_HEADER_TESTS ${ESVM_HEADER_TESTS} ${ESVM_INCLUDE_DIRS}/esvmCreateSampleFiles.h)
    set(ESVM_HEADER_TESTS ${ESVM_HEADER_TESTS} ${ESVM_INCLUDE_DIRS}/esvmTests.h)
    set(ESVM_SOURCE_TESTS ${ESVM_SOURCE_TESTS} ${ESVM_SOURCES_DIRS}/esvmCreateSampleFiles.cpp)
    set(ESVM_SOURCE_TESTS ${ESVM_SOURCE_TESTS} ${ESVM_SOURCES_DIRS}/esvmTests.cpp)
    set(ESVM_SOURCE_TESTS ${ESVM_SOURCE_TESTS} ${ESVM_SOURCES_DIRS}/main.cpp)
endif()

# ESVM configs
set(ESVM_INCLUDE_DIR        ${ESVM_INCLUDE_DIRS})
set(ESVM_LIBRARY_DEBUG      ${ESVM_PROJECT}${CMAKE_DEBUG_POSTFIX}${CMAKE_${ESVM_LIBRARY_TYPE}_LIBRARY_SUFFIX})
set(ESVM_LIBRARY_RELEASE    ${ESVM_PROJECT}${CMAKE_RELEASE_POSTFIX}${CMAKE_${ESVM_LIBRARY_TYPE}_LIBRARY_SUFFIX})

# ESVM root path (esvmOptions)
add_definitions(-DESVM_ROOT_PATH="${ESVM_ROOT_DIR}")

# find FeatureExtractorHOG (feHOG)
if (${ESVM_BUILD_HOG})
    message(STATUS "Trying to find FeatureExtractorHOG (feHOG) to build from source ('ESVM_BUILD_HOG'=ON)")
    find_package(FeatureExtractorHOG REQUIRED MODULE)
    add_subdirectory(${feHOG_ROOT_DIR} ${ESVM_LIBRARY_DIR})    # will raise error if invalid dir
    if (${feHOG_FOUND})
        set(ESVM_INCLUDE_DIRS ${ESVM_INCLUDE_DIRS}  ${feHOG_INCLUDE_DIRS})
        set(ESVM_HEADER_FILES ${ESVM_HEADER_FILES}  ${feHOG_HEADER_FILES})
        set(ESVM_SOURCES_DIRS ${ESVM_SOURCES_DIRS}  ${feHOG_SOURCE_DIR})
        set(ESVM_SOURCE_FILES ${ESVM_SOURCE_FILES}  ${feHOG_SOURCE_FILES})
        # display variables required for building from source
        foreach(var IMPLEMENTATION WITH_OPENMP BUILD_TESTS LIBRARY_TYPE)
            mark_as_advanced(CLEAR feHOG_${var})
        endforeach()
    endif()
    # hide variables required for building from source
    mark_as_advanced(FORCE ${feHOG_DIR})
else()
    # find feHOG installed config
    find_package(feHOG REQUIRED CONFIG
                 NAMES feHOG feHOG FE_HOG FeatureExtractorHOG
                 PATHS ${feHOG_ROOT_DIR} ${feHOG_ROOT_DIR}/cmake ${feHOG_DIR} ${feHOG_DIR}/cmake)
    set(ESVM_INCLUDE_DIRS   ${ESVM_INCLUDE_DIRS}    ${feHOG_INCLUDE_DIRS})
    set(ESVM_HEADER_FILES   ${ESVM_HEADER_FILES}    ${feHOG_HEADER_FILES})
    set(ESVM_LIBRARIES      ${ESVM_LIBRARIES}       $<$<CONFIG:Debug>:${feHOG_LIBRARY_DEBUG}>$<$<CONFIG:Release>:${feHOG_LIBRARY_RELEASE}>)
    set(ESVM_LIBRARY_DIRS   ${ESVM_LIBRARY_DIRS}    ${feHOG_LIBRARY_DIRS})
    # hide variables required for building from source
    foreach(var IMPLEMENTATION WITH_OPENMP BUILD_TESTS LIBRARY_TYPE)
        mark_as_advanced(CLEAR feHOG_${var})
    endforeach()
    # display variable set with 'config' mode with link to installed library
    mark_as_advanced(FORCE ${feHOG_DIR})
endif()

# find FeatureExtractorLBP (feLBP)
if (${ESVM_USE_LBP})
    find_package(feLBP)
    add_subdirectory(${FELBP_ROOT_DIR} ${ESVM_LIBRARY_DIR})    # will raise error if invalid dir
    if (${FELBP_FOUND})
        set(ESVM_INCLUDE_DIRS ${ESVM_INCLUDE_DIRS} ${FELBP_INCLUDE_DIR})
        set(FELBP_FOUND ${feLBP_FOUND})
        add_definitions(-DESVM_HAS_FELBP)
    endif()
endif()

# find Common(C++)
find_package(CommonCpp REQUIRED CONFIG
             NAMES "CommonCpp" "Common-Cpp" "Common_Cpp" "CommonC++" "Common-C++" "Common_C++"
             PATHS ${COMMON_CPP_ROOT_DIR} ${COMMON_CPP_ROOT_DIR}/cmake ${CommonCpp_DIR} ${CommonCpp_DIR}/cmake)
if (${CommonCpp_FOUND})
    set(ESVM_INCLUDE_DIRS   ${ESVM_INCLUDE_DIRS}    ${CommonCpp_INCLUDE_DIRS})
    set(ESVM_LIBRARIES      ${ESVM_LIBRARIES}       $<$<CONFIG:Debug>:${CommonCpp_LIBRARY_DEBUG}>$<$<CONFIG:Release>:${CommonCpp_LIBRARY_RELEASE}>)
    set(ESVM_LIBRARY_DIRS   ${ESVM_LIBRARY_DIRS}    ${CommonCpp_LIBRARY_DIRS})
endif()

# find Boost
option(Boost_USE_STATIC_LIBS    OFF)
option(Boost_USE_MULTITHREADED  ON)
option(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost REQUIRED COMPONENTS filesystem system)
if (${Boost_FOUND})
    set(ESVM_INCLUDE_DIRS   ${ESVM_INCLUDE_DIRS}    ${Boost_INCLUDE_DIRS})
    set(ESVM_LIBRARIES      ${ESVM_LIBRARIES}       ${Boost_LIBRARIES})
    set(ESVM_LIBRARY_DIRS   ${ESVM_LIBRARY_DIRS}    ${Boost_LIBRARY_DIRS})
endif()

# find OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# find OpenCV
find_package(OpenCV 3 REQUIRED)
set(ESVM_INCLUDE_DIRS   ${ESVM_INCLUDE_DIRS}    ${OpenCV_INCLUDE_DIRS})
set(ESVM_LIBRARIES      ${ESVM_LIBRARIES}       ${OpenCV_LIBRARIES})

# find LIBSVM
find_package(LibSVM REQUIRED)
set(ESVM_INCLUDE_DIRS   ${ESVM_INCLUDE_DIRS}    ${LIBSVM_INCLUDE_DIRS})
if (${ESVM_BUILD_LIBSVM})
    set(ESVM_SOURCE_FILES ${ESVM_SOURCE_FILES}  ${LIBSVM_SOURCE_FILE})
    set(ESVM_HEADER_FILES ${ESVM_HEADER_FILES}  ${LIBSVM_HEADER_FILE})
    if (${ESVM_BUILD_TESTS})
        set(ESVM_SOURCE_TESTS ${ESVM_SOURCE_TESTS}  ${LIBSVM_SOURCE_FILE})
        set(ESVM_HEADER_TESTS ${ESVM_HEADER_TESTS}  ${LIBSVM_HEADER_FILE})
    endif()
else()
    set(ESVM_LIBRARIES  ${ESVM_LIBRARIES}       ${LIBSVM_LIBRARIES})
endif()

# find mvector
find_package(mvector REQUIRED)
set(ESVM_INCLUDE_DIRS   ${ESVM_INCLUDE_DIRS}    ${MVECTOR_INCLUDE_DIR})

# include/linking
set(ESVM_LIBRARY_NAME ${ESVM_PROJECT})
add_library(${ESVM_LIBRARY_NAME} ${ESVM_SOURCE_FILES} ${ESVM_HEADER_FILES})
if(MSVC)
    set_target_properties(${ESVM_LIBRARY_NAME} PROPERTIES LINKER_LANGUAGE C++)
endif()
target_include_directories(${ESVM_LIBRARY_NAME} PUBLIC ${ESVM_INCLUDE_DIRS})
target_link_libraries(${ESVM_LIBRARY_NAME} ${ESVM_LIBRARIES})
link_directories(${ESVM_LIBRARY_DIRS})

# build tests
if (${ESVM_BUILD_TESTS})
    set(ESVM_TESTS ${ESVM_PROJECT}_Test)
    if (${ESVM_BUILD_TESTS_MODE} STREQUAL ${ESVM_BUILD_LIBRARY})
        add_definitions(-DFEATURE_EXTRACTOR_HOG_USE_IMPL)
        add_executable(${ESVM_TESTS} ${ESVM_SOURCE_TESTS} ${ESVM_HEADER_TESTS})
        target_link_libraries(${ESVM_TESTS} ${ESVM_LIBRARIES} ${ESVM_LIBRARY_NAME})
    else()
        add_executable(${ESVM_TESTS} ${ESVM_SOURCE_TESTS} ${ESVM_HEADER_TESTS} ${ESVM_SOURCE_FILES} ${ESVM_HEADER_FILES})
        target_link_libraries(${ESVM_TESTS} ${ESVM_LIBRARIES})
    endif()
    target_include_directories(${ESVM_TESTS} PUBLIC ${ESVM_INCLUDE_DIRS})
endif()

# fix config paths as required
string(REGEX REPLACE "\\\\" "/" INSTALL_INCLUDE_DIR ${INSTALL_INCLUDE_DIR}) 
string(REGEX REPLACE "\\\\" "/" INSTALL_BINARY_DIR  ${INSTALL_BINARY_DIR}) 
string(REGEX REPLACE "\\\\" "/" INSTALL_LIBRARY_DIR ${INSTALL_LIBRARY_DIR}) 

# update config file
set(ESVM_CONFIG_INC_DIR ${INSTALL_INCLUDE_DIR})
set(ESVM_CONFIG_BIN_DIR ${INSTALL_BINARY_DIR})
set(ESVM_CONFIG_LIB_DIR ${INSTALL_LIBRARY_DIR})
set(ESVM_CONFIG_LIB_DBG ${INSTALL_LIBRARY_DIR}/${ESVM_LIBRARY_DEBUG})
set(ESVM_CONFIG_LIB_REL ${INSTALL_LIBRARY_DIR}/${ESVM_LIBRARY_RELEASE})
set(ESVM_CONFIG_FILE    ${ESVM_PROJECT}Config.cmake)
if (${CMAKE_BINARY_DIR} STREQUAL ${CMAKE_SOURCE_DIR})
    set(ESVM_CONFIG_PATH ${ESVM_CONFIG_FILE})
else()
    set(ESVM_CONFIG_PATH ${CMAKE_BINARY_DIR}/${ESVM_CONFIG_FILE})
endif()
file(RELATIVE_PATH REL_INCLUDE_DIR "${INSTALL_CMAKE_DIR}" "${INSTALL_INCLUDE_DIR}")
configure_file(cmake/${ESVM_CONFIG_FILE}.in ${ESVM_CONFIG_PATH})

# install
install(TARGETS ${ESVM_LIBRARY_NAME}
        RUNTIME DESTINATION ${INSTALL_BINARY_DIR}
        LIBRARY DESTINATION ${INSTALL_LIBRARY_DIR}
        ARCHIVE DESTINATION ${INSTALL_LIBRARY_DIR})
install(FILES     ${ESVM_HEADER_FILES} DESTINATION ${INSTALL_INCLUDE_DIR})
install(FILES     ${ESVM_CONFIG_PATH}  DESTINATION ${INSTALL_CMAKE_DIR})
if (${ESVM_BUILD_TESTS})
    install(TARGETS ${ESVM_TESTS} RUNTIME DESTINATION ${INSTALL_BINARY_DIR})
endif()
