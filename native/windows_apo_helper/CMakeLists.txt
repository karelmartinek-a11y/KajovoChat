cmake_minimum_required(VERSION 3.20)

project(kajovochat_windows_apo LANGUAGES CXX)

enable_testing()
option(BUILD_TESTING "Build tests" ON)

add_library(kajovochat_windows_apo SHARED
    src/kajovochat_windows_apo.cpp
    src/kajovochat_windows_apo_com.cpp
    exports.def
)

target_compile_features(kajovochat_windows_apo PRIVATE cxx_std_17)
target_include_directories(kajovochat_windows_apo PRIVATE include ../windows_aec_helper/include ../windows_aec_helper/src)
target_compile_definitions(kajovochat_windows_apo PRIVATE
    KAJOVOCHAT_WINDOWS_APO_BUILD_DLL
    KAJOVOCHAT_WINDOWS_AEC_BUILD_DLL
    WIN32_LEAN_AND_MEAN
    NOMINMAX
)

if (MSVC)
    target_compile_options(kajovochat_windows_apo PRIVATE /W4 /permissive-)
else()
    target_compile_options(kajovochat_windows_apo PRIVATE -Wall -Wextra -Wpedantic)
endif()

set_target_properties(kajovochat_windows_apo PROPERTIES
    OUTPUT_NAME "kajovochat_windows_apo"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/$<CONFIG>"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/$<CONFIG>"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/$<CONFIG>"
)

if (BUILD_TESTING)
    add_executable(kajovochat_windows_apo_test
        tests/windows_apo_smoke_test.cpp
    )
    target_compile_features(kajovochat_windows_apo_test PRIVATE cxx_std_17)
    target_include_directories(kajovochat_windows_apo_test PRIVATE include ../windows_aec_helper/include ../windows_aec_helper/src)
    target_link_libraries(kajovochat_windows_apo_test PRIVATE kajovochat_windows_apo)
    set_target_properties(kajovochat_windows_apo_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/$<CONFIG>"
    )

    add_test(NAME windows_apo_smoke_test COMMAND $<TARGET_FILE:kajovochat_windows_apo_test>)
    set_tests_properties(windows_apo_smoke_test PROPERTIES
        WORKING_DIRECTORY "$<TARGET_FILE_DIR:kajovochat_windows_apo_test>"
        ENVIRONMENT "PATH=$<TARGET_FILE_DIR:kajovochat_windows_apo_test>;$ENV{PATH}"
    )

    add_executable(kajovochat_windows_apo_com_test
        tests/windows_apo_com_smoke_test.cpp
    )
    target_compile_features(kajovochat_windows_apo_com_test PRIVATE cxx_std_17)
    target_include_directories(kajovochat_windows_apo_com_test PRIVATE include ../windows_aec_helper/include ../windows_aec_helper/src)
    target_link_libraries(kajovochat_windows_apo_com_test PRIVATE kajovochat_windows_apo)
    set_target_properties(kajovochat_windows_apo_com_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/$<CONFIG>"
    )
    add_test(NAME windows_apo_com_smoke_test COMMAND $<TARGET_FILE:kajovochat_windows_apo_com_test>)
    set_tests_properties(windows_apo_com_smoke_test PROPERTIES
        WORKING_DIRECTORY "$<TARGET_FILE_DIR:kajovochat_windows_apo_com_test>"
        ENVIRONMENT "PATH=$<TARGET_FILE_DIR:kajovochat_windows_apo_com_test>;$ENV{PATH}"
    )
endif()
