BUILD_EXAMPLES=OFF
BUILD_TYPE=Release
CMAKE_ARGS:=$(CMAKE_ARGS)

default:
	@mkdir -p build
	@cd build && cmake .. -DBUILD_EXAMPLES=$(BUILD_EXAMPLES) \
                              -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
                              -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
                              $(CMAKE_ARGS)
	@cd build && make

debug:
	@make default BUILD_TYPE=Debug

apps:
	@make default BUILD_EXAMPLES=ON

debug_apps:
	@make debug BUILD_EXAMPLES=ON

clean:
	@rm -rf build*
