#========================================
# ExemplarSVM-LIBSVM 
#========================================

cmake_minimum_required(VERSION 2.8)

# project
set(ESVM_PROJECT "ExemplarSVM-LIBSVM")
project(${ESVM_PROJECT})

if(MSVC)	
	set(CMAKE_USE_RELATIVE_PATHS ON CACHE INTERNAL "" FORCE)
endif()

# set ESVM paths
set(ESVM_ROOT_DIR 		"${CMAKE_CURRENT_LIST_DIR}")
set(ESVM_EXECUTABLE_DIR "${ESVM_ROOT_DIR}/bld")
set(ESVM_SOURCES_DIR 	"${ESVM_ROOT_DIR}/src")
set(ESVM_INCLUDE_DIR 	"${ESVM_ROOT_DIR}/inc")

# include cmake subdirectory to find packages
list(APPEND CMAKE_MODULE_PATH "${ESVM_ROOT_DIR}/cmake")

# find ESVM header/source files
file(GLOB ESVM_SOURCE_FILES "${ESVM_SOURCES_DIR}/*.cpp" "${ESVM_SOURCES_DIR}/*.c")
file(GLOB ESVM_HEADER_FILES "${ESVM_INCLUDE_DIR}/*.hpp" "${ESVM_INCLUDE_DIR}/*.h")

# ESVM options
option(ESVM_USE_LBP "Employ FeatureExtractorLBP descriptor for ExemplarSVM" OFF)

# find FeatureExtractorHOG
find_package(FeatureExtractorHOG REQUIRED)
if (${FEHOG_FOUND})
	set(ESVM_INCLUDE_DIR 	${ESVM_INCLUDE_DIR} 	${FEHOG_INCLUDE_DIR})
	set(ESVM_HEADER_FILES	${ESVM_HEADER_FILES}	${FEHOG_HEADER_FILES})
	set(ESVM_SOURCES_DIR	${ESVM_SOURCES_DIR}		${FEHOG_SOURCE_DIR})
	set(ESVM_SOURCE_FILES} 	${ESVM_SOURCE_FILES} 	${FEHOG_SOURCE_FILES})	
endif()
set(FEHOG_ROOT_DIR ${FEHOG_ROOT_DIR})
message(".................")
message("${ESVM_SOURCES_DIR}")
message("${ESVM_SOURCE_FILES}")
message("${FEHOG_SOURCE_DIR}")
message("${FEHOG_SOURCE_FILES}")
message(".................")

# find FeatureExtractorLBP
if (${ESVM_USE_LBP})
	find_package(FeatureExtractorLBP)
	set(ESVM_INCLUDE_DIR ${ESVM_INCLUDE_DIR} ${FELBP_INCLUDE_DIR})
	set(FELBP_FOUND ${FeatureExtractorLBP_FOUND})
	add_definitions(-DESVM_HAS_FELBP)
endif()

# find Boost
option(Boost_USE_STATIC_LIBS	ON)
option(Boost_USE_MULTITHREADED  ON)
option(Boost_USE_STATIC_RUNTIME	OFF)
find_package(Boost REQUIRED COMPONENTS filesystem)
if (${Boost_FOUND})
	set(ESVM_INCLUDE_DIR 	${ESVM_INCLUDE_DIR} ${Boost_INCLUDE_DIRS})
	set(ESVM_LIBRARIES 		${ESVM_LIBRARIES} 	${Boost_LIBRARIES})
endif()

# find OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
	
# find OpenCV
find_package(OpenCV 3 REQUIRED)
set(ESVM_INCLUDE_DIR 	${ESVM_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS})
set(ESVM_LIBRARIES 		${ESVM_LIBRARIES} 	${OpenCV_LIBRARIES})

# find LIBSVM
find_package(LibSVM REQUIRED)
set(ESVM_INCLUDE_DIR 	${ESVM_INCLUDE_DIR} ${LIBSVM_INCLUDE_DIRS})
set(ESVM_LIBRARIES 		${ESVM_LIBRARIES} 	${LIBSVM_LIBRARIES})

# find mvector
find_package(mvector REQUIRED)
set(ESVM_INCLUDE_DIR 	${ESVM_INCLUDE_DIR} ${MVECTOR_INCLUDE_DIR})

# include/linking
add_executable(${ESVM_PROJECT} ${ESVM_SOURCE_FILES} ${ESVM_HEADER_FILES})
set_target_properties(${ESVM_PROJECT} PROPERTIES LINKER_LANGUAGE C++)
include_directories(${ESVM_INCLUDE_DIR})
target_link_libraries(${ESVM_PROJECT} ${ESVM_LIBRARIES})
