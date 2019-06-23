CC = clang
CFLAGS = -Wall -O3 -mavx2 -march=native -std=gnu11 -g
LIBS =

SRC=$(wildcard *.c)
HDR=$(wildcard *.h)

test: $(SRC) $(HDR)
	$(CC) -o $@ $(SRC) $(CFLAGS) $(LIBS)

clean:
	rm -f test
