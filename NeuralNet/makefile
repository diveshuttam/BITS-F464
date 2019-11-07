CC = g++
CCFLAGS = -g -Wall -Wpedantic
DEFINES = __NO_DEBUG  


DEFINE_ARGS = $(addprefix -D ,$(DEFINES))
GCC_CMD = $(CC) $(CCFLAGS) $(DEFINE_ARGS) 

MAIN_FILE = ./driver.cpp
ALL_CFILES = nn.cpp training.cpp
ALL_OFILES = $(addsuffix .o, $(basename $(ALL_CFILES)))

all: main

%.o: %.cpp $(ALL_CFILES)
	$(GCC_CMD) -c -o $@ $<

main: $(MAIN_FILE) $(ALL_OFILES)
	$(GCC_CMD) $(MAIN_FILE) nn.o -o driver

clean:
	rm  ./nn.o nn
