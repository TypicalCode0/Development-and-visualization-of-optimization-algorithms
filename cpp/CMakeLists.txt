add_library(function STATIC function.cpp)
add_executable(gd gradient_descent.cpp)
target_link_libraries(gd function)