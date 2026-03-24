cmake_minimum_required(VERSION 3.20)

project(kajovochat_windows_aec LANGUAGES CXX)

enable_testing()

add_library(kajovochat_windows_aec SHARED
    src/kajovochat_windows_aec.cpp
)

target_compile_features(kajovochat_windows_aec PRIVATE cxx_std_17)
target_include_directories(kajovochat_windows_aec PRIVATE include)
target_compile_definitions(kajovochat_windows_aec PRIVATE
    KAJOVOCHAT_WINDOWS_AEC_BUILD_DLL
    WIN32_LEAN_AND_MEAN
    NOMINMAX
)

if (MSVC)
    target_compile_options(kajovochat_windows_aec PRIVATE /W4 /permissive-)
else()
    target_compile_options(kajovochat_windows_aec PRIVATE -Wall -Wextra -Wpedantic)
endif()

set_target_properties(kajovochat_windows_aec PROPERTIES
    OUTPUT_NAME "kajovochat_windows_aec"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/$<CONFIG>"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/$<CONFIG>"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/$<CONFIG>"
)

if (BUILD_TESTING)
    add_executable(kajovochat_windows_aec_test
        tests/windows_aec_smoke_test.cpp
    )
    target_compile_features(kajovochat_windows_aec_test PRIVATE cxx_std_17)
    target_include_directories(kajovochat_windows_aec_test PRIVATE include)
    target_link_libraries(kajovochat_windows_aec_test PRIVATE kajovochat_windows_aec)
    set_target_properties(kajovochat_windows_aec_test PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/$<CONFIG>"
    )

    add_test(NAME windows_aec_smoke_test COMMAND $<TARGET_FILE:kajovochat_windows_aec_test>)
    set_tests_properties(windows_aec_smoke_test PROPERTIES
        WORKING_DIRECTORY "$<TARGET_FILE_DIR:kajovochat_windows_aec_test>"
        ENVIRONMENT "PATH=$<TARGET_FILE_DIR:kajovochat_windows_aec_test>;$ENV{PATH}"
    )
endif()
