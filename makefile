CC = mpicc
CFLAGS = -Wall -g -O3 -march=native -std=c99 -Wno-unused-result -Wno-stringop-overflow
# Try to detect and link BLAS library
ifeq ($(shell uname),Darwin)
    # macOS: Use Accelerate Framework
    LDFLAGS = -lm -framework Accelerate
    CFLAGS += -DHAVE_CBLAS_H
else
    # Linux: Try to find BLAS library
    ifneq ($(shell pkg-config --exists openblas; echo $$?),0)
        ifneq ($(shell pkg-config --exists blas; echo $$?),0)
            # Fallback: use custom BLAS (no external library)
            LDFLAGS = -lm
            CFLAGS += -DUSE_CUSTOM_BLAS
        else
            # Standard BLAS found
            LDFLAGS = -lm $(shell pkg-config --libs blas)
            CFLAGS += -DUSE_CUSTOM_BLAS
        endif
    else
        # Check if OpenBLAS has CBLAS headers
        ifneq ($(shell test -f /usr/include/cblas.h || test -f /usr/local/include/cblas.h; echo $$?),0)
            # No CBLAS headers, use FORTRAN BLAS or fallback
            LDFLAGS = -lm $(shell pkg-config --libs openblas)
            CFLAGS += -DUSE_CUSTOM_BLAS
        else
            # CBLAS headers available
            LDFLAGS = -lm $(shell pkg-config --libs openblas)
            CFLAGS += -DHAVE_CBLAS_H
        endif
    endif
endif
TARGET = gmres

SOURCES = $(wildcard *.c)


OBJECTS = $(patsubst %.c, %.o, $(SOURCES))


.PHONY: all
all: $(TARGET)

$(TARGET): $(OBJECTS)
	@echo "==> Linking to create executable: $@"
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

%.o: %.c
	@echo "==> Compiling source file: $<"
	$(CC) $(CFLAGS) -c -o $@ $<

.PHONY: clean
clean:
	@echo "==> Cleaning up generated files..."
	rm -f $(TARGET) $(OBJECTS)