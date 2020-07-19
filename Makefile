BUILD=build
UTEST=OFF
APPS=OFF
CMAKE_ARGS:=$(CMAKE_ARGS)

all:
	@mkdir -p $(BUILD)
	@cd $(BUILD); cmake .. -DBUILD_APPS=$(APPS) -DCMAKE_BUILD_TYPE=Release $(CMAKE_ARGS) && $(MAKE)
	@echo -e "\n Now do 'make install' to install this package.\n"

debug:
	@mkdir -p $(BUILD)
	@cd $(BUILD); cmake .. -DBUILD_APPS=$(APPS) -DCMAKE_BUILD_TYPE=Debug $(CMAKE_ARGS) && $(MAKE)

apps:
	@$(MAKE) all APPS=ON

debug_apps:
	@$(MAKE) debug APPS=ON

clean:
	@rm -rf $(BUILD)
