CC = g++
CCFLAGS = -g -Wall -Wpedantic
DEFINES = __NO_DEBUG __MY_OWN_DATA_ELE 

MAIN_FILE = ./driver.cpp

DEFINE_ARGS = $(addprefix -D ,$(DEFINES))
GCC_CMD = $(CC) $(CCFLAGS) $(DEFINE_ARGS) 

all: objects main

objects: nn.cpp 
	$(GCC_CMD) nn.cpp -c -o nn.o

main: $(MAIN_FILE)
	$(GCC_CMD) $(MAIN_FILE) nn.o -o driver

clean:
	rm  ./nn.o nn
