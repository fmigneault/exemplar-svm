# - Find LIBLINEAR
# LIBLINEAR is a Library for Support Vector Machines
# available at https://github.com/cjlin1/liblinear
#
# This file based on http://www.cmake.org/Wiki/CMakeUserFindLibSVM
# Modified for LIBLINEAR by Francis Charette Migneault
#
# The module defines the following variables:
#  LIBLINEAR_FOUND - the system has LIBLINEAR
#  LIBLINEAR_INCLUDE_DIR - where to find svm.h
#  LIBLINEAR_INCLUDE_DIRS - LIBLINEAR includes
#  LIBLINEAR_LIBRARY - where to find the LIBLINEAR library
#  LIBLINEAR_LIBRARIES - aditional libraries
#  LIBLINEAR_MAJOR_VERSION - major version
#  LIBLINEAR_MINOR_VERSION - minor version
#  LIBLINEAR_PATCH_VERSION - patch version
#  LIBLINEAR_VERSION_STRING - version (ex. 2.9.0)
#  LIBLINEAR_ROOT_DIR - root dir (ex. /usr/local)

#=============================================================================
# Copyright 2000-2009 Kitware, Inc., Insight Software Consortium
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
# 
#   * Redistributions of source code must retain the above copyright notice,
#     this list of conditions and the following disclaimer.
# 
#   * Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
# 
#   * Neither the names of Kitware, Inc., the Insight Software Consortium, nor
#     the names of their contributors may be used to endorse or promote products
#     derived from this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#=============================================================================

set(LIBLINEAR_ROOT_DIR ${LIBLINEAR_ROOT_DIR} CACHE PATH "Root directory of LIBLINEAR")

# set LIBLINEAR_INCLUDE_DIR
find_path(LIBLINEAR_INCLUDE_DIR
          NAMES svm.h
          PATHS ${LIBLINEAR_ROOT_DIR} ENV LIBLINEAR_ROOT_DIR
          PATH_SUFFIXES
            include
            LIBLINEAR
          DOC   "LIBLINEAR include directory"
)

# set LIBLINEAR_INCLUDE_DIRS
if ( LIBLINEAR_INCLUDE_DIR )
  set ( LIBLINEAR_INCLUDE_DIRS ${LIBLINEAR_INCLUDE_DIR} )
endif()

# set header/source files
set(LIBLINEAR_HEADER_FILE ${LIBLINEAR_INCLUDE_DIR}/svm.h)
set(LIBLINEAR_SOURCE_FILE ${LIBLINEAR_INCLUDE_DIR}/svm.cpp)

# version
set ( _VERSION_FILE ${LIBLINEAR_INCLUDE_DIR}/svm.h )
if ( EXISTS ${_VERSION_FILE} )
  # LIBLINEAR_VERSION_STRING macro defined in svm.h since version 2.8.9
  file ( STRINGS ${_VERSION_FILE} _VERSION_STRING REGEX ".*define[ ]+LIBLINEAR_VERSION[ ]+[0-9]+.*" )
  if ( _VERSION_STRING )
    string ( REGEX REPLACE ".*LIBLINEAR_VERSION[ ]+([0-9]+)" "\\1" _VERSION_NUMBER "${_VERSION_STRING}" )
    math ( EXPR LIBLINEAR_MAJOR_VERSION "${_VERSION_NUMBER} / 100" )
    math ( EXPR LIBLINEAR_MINOR_VERSION "(${_VERSION_NUMBER} % 100 ) / 10" )
    math ( EXPR LIBLINEAR_PATCH_VERSION "${_VERSION_NUMBER} % 10" )
    set ( LIBLINEAR_VERSION_STRING "${LIBLINEAR_MAJOR_VERSION}.${LIBLINEAR_MINOR_VERSION}.${LIBLINEAR_PATCH_VERSION}" )
  endif ()
endif ()

# check version
set ( _LIBLINEAR_VERSION_MATCH TRUE )
if ( LIBLINEAR_FIND_VERSION AND LIBLINEAR_VERSION_STRING )
  if ( LIBLINEAR_FIND_VERSION_EXACT )
    if ( NOT ${LIBLINEAR_FIND_VERSION} VERSION_EQUAL ${LIBLINEAR_VERSION_STRING} )
      set ( _LIBLINEAR_VERSION_MATCH FALSE )
    endif ()
  else ()
    if ( ${LIBLINEAR_FIND_VERSION} VERSION_GREATER ${LIBLINEAR_VERSION_STRING} )
      set ( _LIBLINEAR_VERSION_MATCH FALSE )
    endif ()
  endif ()
endif ()

# set LIBLINEAR_LIBRARY
find_library(LIBLINEAR_LIBRARY
             NAMES          svm linear
             PATHS          ${LIBLINEAR_ROOT_DIR}
             PATH_SUFFIXES  lib
             DOC            "LIBLINEAR library location"
)
get_filename_component(LIBLINEAR_FOUND_FILE_NAME ${LIBLINEAR_LIBRARY} NAME)
if (NOT (LIBLINEAR_FOUND_FILE_NAME AND LIBLINEAR_LIBRARY))
    if (UNIX)
        find_library(LIBLINEAR_LIBRARY
                     NAMES  liblinear.so liblinear.so.2 
                     PATHS  ${LIBLINEAR_ROOT_DIR}/windows
                     DOC    "LIBLINEAR library location"
        )
    elseif (WIN32)
        find_library(LIBLINEAR_LIBRARY
                     NAMES  liblinear
                     PATHS  ${LIBLINEAR_ROOT_DIR}/windows
                     DOC    "LIBLINEAR library location"
        )
    endif()
endif()

# set LIBLINEAR_LIBRARIES
set(LIBLINEAR_LIBRARIES ${LIBLINEAR_LIBRARY})

# link with math library on unix
if ( UNIX )
    find_library(M_LIB m)
    mark_as_advanced(M_LIB)
    list (APPEND LIBLINEAR_LIBRARIES ${M_LIB} )
endif()

# try to guess root dir from include dir
if (LIBLINEAR_INCLUDE_DIR)
  string ( REGEX REPLACE "(.*)/include.*" "\\1" LIBLINEAR_ROOT_DIR ${LIBLINEAR_INCLUDE_DIR} )
# try to guess root dir from library dir
elseif (LIBLINEAR_LIBRARY)
  string ( REGEX REPLACE "(.*)/lib[/|32|64].*" "\\1" LIBLINEAR_ROOT_DIR ${LIBLINEAR_LIBRARY} )
endif ()

# handle REQUIRED and QUIET options
include( FindPackageHandleStandardArgs )
find_package_handle_standard_args ( LIBLINEAR DEFAULT_MSG
    _LIBLINEAR_VERSION_MATCH
    LIBLINEAR_SOURCE_FILE
    LIBLINEAR_HEADER_FILE
    LIBLINEAR_LIBRARY
    LIBLINEAR_INCLUDE_DIR
    LIBLINEAR_INCLUDE_DIRS
    LIBLINEAR_LIBRARIES
    LIBLINEAR_ROOT_DIR
)

mark_as_advanced(
    LIBLINEAR_SOURCE_FILE
    LIBLINEAR_HEADER_FILE
    LIBLINEAR_LIBRARY
    LIBLINEAR_LIBRARIES
    LIBLINEAR_INCLUDE_DIR
    LIBLINEAR_INCLUDE_DIRS
    LIBLINEAR_ROOT_DIR
    LIBLINEAR_VERSION_STRING
    LIBLINEAR_MAJOR_VERSION
    LIBLINEAR_MINOR_VERSION
    LIBLINEAR_PATCH_VERSION
)
