add_library(planning INTERFACE include/planning/env.h)
set_target_properties(planning PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(planning INTERFACE ${PROJECT_ROOT_DIR}/planning/include)
target_link_libraries(planning INTERFACE Fastor)
target_include_directories(planning INTERFACE ${PCG_INCLUDE_DIRS})
target_include_directories(planning INTERFACE ${DBG_MACRO_INCLUDE_DIRS})

if(SKBUILD)
  nanobind_add_module(planning_ext NB_STATIC include/planning/env.h src/env.cpp)
  target_include_directories(planning_ext PUBLIC ${PROJECT_ROOT_DIR}/planning/include)
  target_link_libraries(planning_ext PUBLIC Fastor)
  target_include_directories(planning_ext PUBLIC ${PCG_INCLUDE_DIRS})
  target_include_directories(planning_ext PUBLIC ${DBG_MACRO_INCLUDE_DIRS})

  install(TARGETS planning_ext LIBRARY DESTINATION .)
endif()

add_executable(main src/main.cpp)
target_link_libraries(main planning)
