add_llvm_loadable_module(enerc
  # Scaffolding.
  transform.cpp
  registration.cpp
  approxinfo.cpp
  log.cpp

  # Optimizations.
  loopparallelize.cpp
  loopperf.cpp
  acceptaa.cpp
  desync.cpp
  npu.cpp
  error.cpp
  functionapprox.cpp
)
set_target_properties( enerc PROPERTIES 
    COMPILE_FLAGS "-fno-rtti -fvisibility-inlines-hidden"
)
