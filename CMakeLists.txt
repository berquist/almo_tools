cmake_minimum_required(VERSION 3.8)
project(almo_tools CXX)

find_package(Armadillo REQUIRED)

set(SRC
    indices.C
    printing.C
    utils.C
    )
# restricted_multiplication_main.C

add_library(almo_tools SHARED ${SRC})
target_link_libraries(almo_tools ${ARMADILLO_LIBRARIES})

set(EXECUTABLES
    restricted_multiplication_main
    )

foreach(EXECUTABLE ${EXECUTABLES})
    add_executable(${EXECUTABLE} ${EXECUTABLE}.C)
    target_link_libraries(${EXECUTABLE} almo_tools)
endforeach()

enable_testing()

set(TESTS
    indexing_test
    )

foreach(TEST ${TESTS})
    add_executable(${TEST} ${TEST}.C)
    # set_target_properties(${TEST} PROPERTIES
    #     COMPILE_FLAGS "${TESTS_COMPILE_FLAGS}"
    #     LINK_FLAGS "${TESTS_LINK_FLAGS}")
    target_link_libraries(${TEST} almo_tools)
    add_test(NAME "${TEST}"
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMAND ${TEST})
endforeach()
