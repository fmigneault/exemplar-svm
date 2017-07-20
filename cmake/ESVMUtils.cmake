# Utility functions and macros for ESVM

include(CMakeParseArguments)

macro(toggle_display_variable var display)
    if (${display})
        mark_as_advanced(CLEAR ${var})  # visible from CMake GUI
    else()
        mark_as_advanced(FORCE ${var})  # hidden from CMake GUI
        unset(${var})   # unavailable for script, still in cache
    endif()
endmacro()