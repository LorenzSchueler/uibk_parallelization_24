cmake_minimum_required(VERSION 3.10)


add_executable (run_full_code
    main_full_code.cpp
    )

target_include_directories(run_full_code
    PUBLIC ${CODE_INCLUDE_DIR}
)

target_link_libraries(run_full_code
    PUBLIC utilities
    PUBLIC sim_setup
    PUBLIC solver
)

install(TARGETS run_full_code
    RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/bin
)

add_executable (run_full_code_parallel
    main_full_code_parallel.cpp
    )

target_include_directories(run_full_code_parallel
    PUBLIC ${CODE_INCLUDE_DIR}
)

target_link_libraries(run_full_code_parallel
    PUBLIC utilities
    PUBLIC sim_setup
    PUBLIC mpi_stuff
    PUBLIC solver
    PUBLIC MPI::MPI_C
)

install(TARGETS run_full_code_parallel
    RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/bin
)



add_executable (run_singlestep
    main_test_singlesteps.cpp
    )

target_include_directories(run_singlestep
    PUBLIC ${CODE_INCLUDE_DIR}
)

target_link_libraries(run_singlestep
    PUBLIC utilities
    PUBLIC sim_setup
    PUBLIC solver
    PUBLIC IO
)

install(TARGETS run_singlestep
    RUNTIME DESTINATION ${PROJECT_SOURCE_DIR}/bin
)