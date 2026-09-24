# Strict warnings for takt's own targets. Third-party code is pulled in as SYSTEM so it stays quiet.
function(takt_set_warnings target)
  if(MSVC)
    target_compile_options(${target} PRIVATE /W4 /permissive- /w14265 /w14296 /w14311
      $<$<BOOL:${TAKT_WARNINGS_AS_ERRORS}>:/WX>)
  else()
    target_compile_options(${target} PRIVATE
      -Wall -Wextra -Wpedantic
      -Wshadow -Wconversion -Wsign-conversion -Wold-style-cast
      -Wnon-virtual-dtor -Woverloaded-virtual -Wcast-align
      -Wnull-dereference -Wdouble-promotion -Wimplicit-fallthrough
      $<$<BOOL:${TAKT_WARNINGS_AS_ERRORS}>:-Werror>)
  endif()
endfunction()
