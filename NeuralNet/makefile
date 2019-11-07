CC = gcc
CCFLAGS = -g -Wall -Wpedantic
DEFINES = __NO_DEBUG __MY_OWN_DATA_ELE 

MAIN_FILE = ./nn.cpp

DEFINE_ARGS = $(addprefix -D ,$(DEFINES))
GCC_CMD = $(CC) $(CCFLAGS) $(DEFINE_ARGS) 

all: compiler

compiler: $(MAIN_FILE)
	$(GCC_CMD) $(MAIN_FILE) -o nn

clean:
	rm  ./nn.o nn