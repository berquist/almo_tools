cmake_minimum_required(VERSION 3.8)
project(almo_tools CXX)

find_package(Armadillo REQUIRED)

find_package(PythonLibs REQUIRED)
include_directories(${PYTHON_INCLUDE_DIRS})

find_package(pybind11)

set(SRC
    example_class.C
    indices.C
    index_printing.C
    utils.C
    )

add_library(almo_tools SHARED ${SRC})
target_link_libraries(almo_tools
    ${ARMADILLO_LIBRARIES}
    # ${PYTHON_LIBRARIES}
    )
# add_library(almo_tools MODULE ${SRC})
# target_link_libraries(almo_tools
#     PRIVATE
#     ${ARMADILLO_LIBRARIES}
#     pybind11::module
#     )
# set_target_properties(almo_tools
#     PROPERTIES
#     PREFIX "${PYTHON_MODULE_PREFIX}"
#     SUFFIX "${PYTHON_MODULE_EXTENSION}")

set(EXECUTABLES
    example_class_main
    restricted_multiplication_main
    )

foreach(EXECUTABLE ${EXECUTABLES})
    add_executable(${EXECUTABLE} ${EXECUTABLE}.C)
    target_link_libraries(${EXECUTABLE} almo_tools)
endforeach()

pybind11_add_module(almo_tools_cxx MODULE ${SRC})
target_link_libraries(almo_tools_cxx PRIVATE ${ARMADILLO_LIBRARIES})

enable_testing()

set(TESTS
    indexing_test
    )

foreach(TEST ${TESTS})
    add_executable(${TEST} ${TEST}.C)
    # Not needed for now.
    # set_target_properties(${TEST} PROPERTIES
    #     COMPILE_FLAGS "${TESTS_COMPILE_FLAGS}"
    #     LINK_FLAGS "${TESTS_LINK_FLAGS}")
    target_link_libraries(${TEST} almo_tools)
    add_test(NAME "${TEST}"
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMAND ${TEST})
endforeach()
