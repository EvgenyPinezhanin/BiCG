file(GLOB samples_list RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} *.cpp)

find_package(OpenMP)

foreach(test_filename ${samples_list})
  get_filename_component(test ${test_filename} NAME_WE)

  add_executable(${test} ${test_filename})
  target_link_libraries(${test} ${PROJECT_LIBRARY})
  if(OpenMP_CXX_FOUND)
      target_link_libraries(${test} OpenMP::OpenMP_CXX)
  endif()
  set_target_properties(${test} PROPERTIES
    OUTPUT_NAME "${test}"
    PROJECT_LABEL "${test}"
    RUNTIME_OUTPUT_DIRECTORY "../"
  )

  # REPORT
  message( STATUS "")
  message( STATUS "General configuration for ${test}")
  message( STATUS "OK")
  message( STATUS "")

endforeach()
