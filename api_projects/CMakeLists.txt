cmake_minimum_required(VERSION 3.25)
project(rest-api)

#find libs

find_package(Crow REQUIRED)

#LİNK SOURCE FİLES

add_executable(${PROJECT_NAME}

src/main.cpp
src/server.cpp
src/handlers/base.cpp
src/handlers/user.cpp

)

#link libs

target_link_libraries(${PROJECT_NAME}
PUBLIC
 Crow::Crow

)

#link headers

target_include_directories(${PROJECT_NAME}
PUBLIC
    ${PROJECT_SOURCE_DIR}/include

)