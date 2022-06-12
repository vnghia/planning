file(GLOB HEADER_FILES include/planning/*.h)
file(GLOB SRC_FILES src/*.cpp)

add_library(planning STATIC ${HEADER_FILES} ${SRC_FILES})

target_include_directories(planning PUBLIC ${PROJECT_ROOT_DIR}/planning/include)
target_link_libraries(planning dbg_macro)
target_link_libraries(planning Fastor)

target_compile_options(planning PRIVATE -Wall -Wextra)
