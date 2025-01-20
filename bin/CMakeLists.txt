if(FF_BUILD_SUBSTITUTION_TOOL)
  add_subdirectory(protobuf-to-json)
endif()

if(FF_BUILD_VISUALIZATION_TOOL)
  add_subdirectory(substitution-to-dot)
endif()

if(FF_BUILD_ARG_PARSER)
  add_subdirectory(arg_parser)
endif()

if(FF_BUILD_BIN_EXPORT_MODEL_ARCH)
  add_subdirectory(export-model-arch)
endif()
