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
    if (CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_definitions(-D_ITERATOR_DEBUG_LEVEL=2 -D_DEBUG)
    else()
        add_definitions(-D_ITERATOR_DEBUG_LEVEL=0)
    endif()
else()
    set(CMAKE_CXX_COMPILER g++)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -fPIC")
endif()

# Offer the user the choice of overriding the installation directories
set(INSTALL_LIB_DIR     lib CACHE PATH "Installation directory for libraries")
set(INSTALL_BIN_DIR     bin CACHE PATH "Installation directory for executables")
set(INSTALL_INCLUDE_DIR inc CACHE PATH "Installation directory for header files")
if(WIN32 AND NOT CYGWIN)
  set(DEF_INSTALL_CMAKE_DIR cmake)
else()
  set(DEF_INSTALL_CMAKE_DIR lib/CMake/${ESVM_PROJECT})
endif()
set(INSTALL_CMAKE_DIR ${DEF_INSTALL_CMAKE_DIR} CACHE PATH "Installation directory for CMake files")

# Make relative paths absolute (needed later on)
foreach(p LIB BIN INCLUDE CMAKE)
  set(var INSTALL_${p}_DIR)
  if(NOT IS_ABSOLUTE "${${var}}")
    set(${var} "${CMAKE_INSTALL_PREFIX}/${${var}}")
  endif()
endforeach()

# set ESVM paths
set(ESVM_ROOT_DIR           "${CMAKE_CURRENT_LIST_DIR}")
set(ESVM_EXECUTABLE_DIR     "${ESVM_ROOT_DIR}/bld")
set(ESVM_SOURCES_DIRS       "${ESVM_ROOT_DIR}/src")
set(ESVM_INCLUDE_DIRS       "${ESVM_ROOT_DIR}/inc")
set(ESVM_LIBRARY_DIR        "${ESVM_ROOT_DIR}/lib")
set(CMAKE_INSTALL_PREFIX    "${ESVM_ROOT_DIR}/install" CACHE STRING "")

# include cmake subdirectory to find packages
list(APPEND CMAKE_MODULE_PATH "${ESVM_ROOT_DIR}/cmake")

# ESVM options
option(ESVM_USE_LBP     "Employ FeatureExtractorLBP descriptor for ExemplarSVM" OFF)
option(ESVM_BUILD_HOG   "Build FeatureExtractorHOG (feHOG) from source"         OFF)
option(ESVM_BUILD_TESTS "Build executable for tests"                            OFF)
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
set(ESVM_INCLUDE_DIR    ${ESVM_INCLUDE_DIRS})
set(ESVM_LIBRARY        ${ESVM_PROJECT}${CMAKE_${ESVM_LIBRARY_TYPE}_LIBRARY_SUFFIX})

# ESVM root path (esvmOptions)
add_definitions(-DESVM_ROOT_PATH="${ESVM_ROOT_DIR}")

# find FeatureExtractorHOG (feHOG)
if (${ESVM_BUILD_HOG})
    unset(${feHOG_DIR} CACHE) # remove variable set with 'config' mode with link to installed library
    message(STATUS "Trying to find FeatureExtractorHOG (feHOG) to build from source ('ESVM_BUILD_HOG'=ON)")
    find_package(FeatureExtractorHOG REQUIRED MODULE)
    add_subdirectory(${FEHOG_ROOT_DIR} ${ESVM_LIBRARY_DIR})    # will raise error if invalid dir
    if (${FEHOG_FOUND})
        set(ESVM_INCLUDE_DIRS ${ESVM_INCLUDE_DIRS}  ${FEHOG_INCLUDE_DIRS})
        set(ESVM_HEADER_FILES ${ESVM_HEADER_FILES}  ${FEHOG_HEADER_FILES})
        set(ESVM_SOURCES_DIRS ${ESVM_SOURCES_DIRS}  ${FEHOG_SOURCE_DIR})
        set(ESVM_SOURCE_FILES ${ESVM_SOURCE_FILES}  ${FEHOG_SOURCE_FILES})
    endif()
else()    
    # remove variables set by building from source    
    foreach(var IMPLEMENTATION WITH_OPENMP BUILD_TESTS LIBRARY_TYPE)
        unset(FEHOG_${var} CACHE)
    endforeach()    
    # find feHOG installed config
    find_package(feHOG REQUIRED CONFIG)
    set(ESVM_INCLUDE_DIRS   ${ESVM_INCLUDE_DIRS}    ${FEHOG_INCLUDE_DIRS})
    set(ESVM_HEADER_FILES   ${ESVM_HEADER_FILES}    ${FEHOG_HEADER_FILES})
    set(ESVM_LIBRARIES      ${ESVM_LIBRARIES}       ${FEHOG_LIBRARIES})
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
find_package(CommonCpp REQUIRED CONFIG)
if (${CommonCpp_FOUND})    
    set(ESVM_INCLUDE_DIRS  ${ESVM_INCLUDE_DIRS} ${CommonCpp_INCLUDE_DIRS})
    set(ESVM_LIBRARIES     ${ESVM_LIBRARIES}    ${CommonCpp_LIBRARIES})
endif()

# find Boost
option(Boost_USE_STATIC_LIBS    OFF)
option(Boost_USE_MULTITHREADED  ON)
option(Boost_USE_STATIC_RUNTIME OFF)
find_package(Boost REQUIRED COMPONENTS filesystem system)
if (${Boost_FOUND})    
    set(ESVM_INCLUDE_DIRS  ${ESVM_INCLUDE_DIRS} ${Boost_INCLUDE_DIRS})
    set(ESVM_LIBRARIES     ${ESVM_LIBRARIES}    ${Boost_LIBRARIES})
    if (WIN32)
        # add library location to allow auto-link as required
        link_directories(${Boost_LIBRARY_DIRS})
        
        # disable auto linking 
        #add_definitions(-DBOOST_ALL_NO_LIB)
        #remove_definitions(-BOOST_NO_EXCEPTIONS)
    endif()
    # if (NOT ${Boost_USE_STATIC_LIBS})
        # # avoid invalid linking error when using shared libs
        # add_definitions(-DBOOST_ALL_DYN_LINK)
    # endif()    
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
set(ESVM_LIBRARIES      ${ESVM_LIBRARIES}       ${LIBSVM_LIBRARIES})

# find mvector
find_package(mvector REQUIRED)
set(ESVM_INCLUDE_DIRS   ${ESVM_INCLUDE_DIRS}  ${MVECTOR_INCLUDE_DIR})

# include/linking
set(ESVM_LIBRARY_NAME ${ESVM_PROJECT})
add_library(${ESVM_LIBRARY_NAME} ${ESVM_SOURCE_FILES} ${ESVM_HEADER_FILES})
if(MSVC)
    set_target_properties(${ESVM_LIBRARY_NAME} PROPERTIES LINKER_LANGUAGE C++)
endif()
target_include_directories(${ESVM_LIBRARY_NAME} PUBLIC ${ESVM_INCLUDE_DIRS})
target_link_libraries(${ESVM_LIBRARY_NAME} ${ESVM_LIBRARIES})

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

# update config file
set(ESVM_CONFIG_INC     ${INSTALL_INCLUDE_DIR})
set(ESVM_CONFIG_LIB     ${INSTALL_LIB_DIR}/${ESVM_LIBRARY})
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
        RUNTIME DESTINATION ${INSTALL_BIN_DIR}
        LIBRARY DESTINATION ${INSTALL_LIB_DIR}
        ARCHIVE DESTINATION ${INSTALL_LIB_DIR})
install(FILES     ${ESVM_HEADER_FILES} DESTINATION ${INSTALL_INCLUDE_DIR})
install(FILES     ${ESVM_CONFIG_PATH}  DESTINATION ${INSTALL_CMAKE_DIR})
if (${ESVM_BUILD_TESTS})
    install(TARGETS ${ESVM_TESTS} RUNTIME DESTINATION ${INSTALL_BIN_DIR})
endif()
