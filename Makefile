.PHONY: all configure build test benchmark test-engine clean rebuild

CMAKE ?= cmake
BUILD_DIR ?= build
SRC_DIR ?= source_code/graphics_engine

all: build

configure:
	$(CMAKE) -S $(SRC_DIR) -B $(BUILD_DIR)

build: configure
	$(CMAKE) --build $(BUILD_DIR) -j

test: build
	ctest --test-dir $(BUILD_DIR) --output-on-failure

test-engine: build
	./$(BUILD_DIR)/test_engine

benchmark: build
	./$(BUILD_DIR)/benchmark

clean:
	$(CMAKE) --build $(BUILD_DIR) --target clean || true
	rm -rf $(BUILD_DIR)

rebuild: clean build
