add_executable(linear linear.cpp)
target_link_libraries(linear planning)
target_compile_options(linear PRIVATE -Wall -Wextra)