CC = mpicc
CFLAGS = -Wall -g -O3 -std=c99 -Wno-unused-result  -Wno-stringop-overflow
LDFLAGS = -lm
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
