add_library(geltif STATIC EXCLUDE_FROM_ALL geltif.cpp)
target_include_directories(geltif PUBLIC ${CMAKE_CURRENT_LIST_DIR})
target_include_directories(geltif PRIVATE
	${CMAKE_CURRENT_LIST_DIR}/../cgltf
	${CMAKE_CURRENT_LIST_DIR}/../stb
	${CMAKE_CURRENT_LIST_DIR}/../howler
	${CMAKE_CURRENT_LIST_DIR}/../glm
)
