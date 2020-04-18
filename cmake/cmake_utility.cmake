function(__build_example example_name public_lib private_lib)
  add_executable(
    ${example_name}
    ${CMAKE_CURRENT_LIST_DIR}/${example_name}.cpp
  )

  target_link_libraries(
    ${example_name}
    PUBLIC
       ${public_lib}
    PRIVATE
       ${private_lib}
  )
endfunction()
