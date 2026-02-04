.PHONY: all configure build test benchmark test-engine clean rebuild

CMAKE ?= cmake
BUILD_DIR ?= build_root

all: build

configure:
	$(CMAKE) -S . -B $(BUILD_DIR)

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
