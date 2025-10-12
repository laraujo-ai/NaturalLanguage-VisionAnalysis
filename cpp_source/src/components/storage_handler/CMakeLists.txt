add_library(storage_handler STATIC src/milvus_storage_handler.cpp)

target_include_directories(storage_handler PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Compile with -fPIC so it can be linked into shared libraries
set_target_properties(storage_handler PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_link_libraries(storage_handler PUBLIC
    milvus_sdk
    common
)
