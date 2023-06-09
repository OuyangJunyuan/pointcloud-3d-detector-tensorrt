macro(set_ifndef_from_env var)
    if (NOT DEFINED ${var})
        if (DEFINED ENV{${var}})
            set(${var} $ENV{${var}})
            message(STATUS "${var}:=\$ENV{${var}}($ENV{${var}})")
        else ()
            message(FATAL_ERROR "neither env variable ${var} nor cmake variable -D${var} is found.")
        endif ()
    endif ()
endmacro()

macro(set_ifndef_from_default var val)
    if (NOT DEFINED ${var})
        set(${var} ${val})
        message(STATUS "${var}:=default(${val})")
    endif ()
endmacro()