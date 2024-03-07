
cc_library(
    name = "board",
    hdrs = ["board.h"],
    srcs = ["board.cc", "added_board_methods.cc"],
)

cc_library(
    name = "player",
    hdrs = ["player.h"],
    srcs = ["player.cc"],
    deps = [
        ":board",
        ":transposition_table",
        ":move_picker",
    ],
)

cc_binary(
    name = "test",
    srcs = ["board_test.cc"],
    deps = [
        ":board",
        ":utils",
    ],
    
)

cc_library(
    name = "utils",
    srcs = ["utils.cc"],
    hdrs = ["utils.h"],
    deps = [
        ":board",
    ],
)

cc_library(
    name = "transposition_table",
    srcs = ["transposition_table.cc"],
    hdrs = ["transposition_table.h"],
    deps = [
        ":board",
    ]
)

cc_library(
    name = "move_picker",
    srcs = ["move_picker.cc"],
    hdrs = ["move_picker.h"],
    deps = [
        ":board",
    ]
)
