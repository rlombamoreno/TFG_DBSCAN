# Makefile for DBSCAN CUDA compilation
# Author: Rodrigo Lomba Moreno
# Institution: Universidad Politécnica de Madrid

# Compiler and flags
NVCC = nvcc
NVCC_FLAGS = -shared -Xcompiler -fPIC -O3
CUDA_ARCH = -arch=sm_70  # Adjust based on your GPU architecture

# Target library
TARGET = libdbscan.so
SOURCE = dbscan.cu

# Default target
all: $(TARGET)

# Compile CUDA code to shared library
$(TARGET): $(SOURCE)
	@echo "Compiling CUDA code..."
	$(NVCC) $(NVCC_FLAGS) $(CUDA_ARCH) -o $(TARGET) $(SOURCE)
	@echo "Library $(TARGET) successfully compiled"

# Clean build artifacts
clean:
	rm -f $(TARGET)
	@echo "Clean completed"

# Install library to system path (optional)
install: $(TARGET)
	cp $(TARGET) /usr/local/lib/
	@echo "Library installed to /usr/local/lib/"

# Debug build
debug: NVCC_FLAGS += -g -G
debug: $(TARGET)

# Show GPU architecture information
info:
	nvidia-smi --query-gpu=compute_cap --format=csv

# Help message
help:
	@echo "DBSCAN CUDA Compilation Makefile"
	@echo "Available targets:"
	@echo "  all      - Compile the library (default)"
	@echo "  clean    - Remove compiled files"
	@echo "  debug    - Compile with debug symbols"
	@echo "  install  - Install library to system path"
	@echo "  info     - Show GPU compute capability"
	@echo "  help     - Show this help message"

.PHONY: all clean install debug info help