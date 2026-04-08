if(NOT DEFINED GELTIF_GLM_PATH)
	message(FATAL_ERROR "GELTIF_GLM_PATH must be set to the path of the glm library")
endif()

add_library(geltif STATIC EXCLUDE_FROM_ALL geltif.cpp)
target_include_directories(geltif PUBLIC
	${CMAKE_CURRENT_LIST_DIR}
	${CMAKE_CURRENT_LIST_DIR}/ext/howler
	${GELTIF_GLM_PATH}
)
target_include_directories(geltif PRIVATE
	${CMAKE_CURRENT_LIST_DIR}/ext/cgltf
	${CMAKE_CURRENT_LIST_DIR}/ext/stb
)
