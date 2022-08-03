add_library(planning INTERFACE include/planning/system.h include/planning/xoshiro.h include/planning/state.h)
set_target_properties(planning PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(planning INTERFACE ${PROJECT_ROOT_DIR}/planning/include)
target_include_directories(planning INTERFACE ${DBG_MACRO_INCLUDE_DIRS})
target_include_directories(planning INTERFACE ${ROBIN_MAP_INCLUDE_DIRS})
target_link_libraries(planning INTERFACE Eigen3::Eigen)

if(SKBUILD)
  nanobind_add_module(planning_ext NB_STATIC include/planning/system.h include/planning/xoshiro.h include/planning/state.h src/system.cpp)
  target_include_directories(planning_ext PUBLIC ${PROJECT_ROOT_DIR}/planning/include)
  target_include_directories(planning_ext PUBLIC ${DBG_MACRO_INCLUDE_DIRS})
  target_include_directories(planning_ext PUBLIC ${ROBIN_MAP_INCLUDE_DIRS})
  target_link_libraries(planning_ext PUBLIC Eigen3::Eigen)

  install(TARGETS planning_ext LIBRARY DESTINATION .)
endif()

add_executable(main src/main.cpp)
target_link_libraries(main planning)
