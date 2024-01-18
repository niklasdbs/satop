
cmake_minimum_required(VERSION 3.16)
project(top_envp)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# Let's nicely support folders in IDEs
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

#add_subdirectory(extern/cnpy)
add_subdirectory(extern/pybind11)

# The compiled library code is here
add_subdirectory(src)


# The executable code is here
add_subdirectory(app)

#python bindings are here
add_subdirectory(python)