CMAKE_MINIMUM_REQUIRED(VERSION 2.8 FATAL_ERROR)
CMAKE_POLICY(VERSION 2.8)
FIND_PACKAGE(Torch REQUIRED)

set(PKGNAME fastrcnn)

file(GLOB_RECURSE luafiles RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "*.lua")

foreach(file ${luafiles})
  get_filename_component(dir ${file} PATH)
  install(FILES ${file} DESTINATION ${Torch_INSTALL_LUA_PATH_SUBDIR}/${PKGNAME}/${dir})
endforeach()

ADD_SUBDIRECTORY(lib)

