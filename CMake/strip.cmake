# https://blog.insane.engineer/post/cmake_strip_binaries/

function(utils_strip TARGET)
    add_custom_command(
        TARGET "${TARGET}" POST_BUILD
        DEPENDS "${TARGET}"
        COMMAND $<$<CONFIG:release>:${CMAKE_STRIP}>
        ARGS --strip-all $<TARGET_FILE:${TARGET}>
    )
endfunction()
