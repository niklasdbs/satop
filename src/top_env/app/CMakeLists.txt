add_executable(app app.cpp)
target_link_libraries(app PRIVATE pybind11::embed ${PYTHON_LIBRARIES})
target_link_libraries(app PRIVATE top_env_lib)
