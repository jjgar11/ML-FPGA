CC     = gcc
CFLAGS = -O2 -Wall -Wextra -std=c11

all: wine_dma_infer

wine_dma_infer: wine_dma_infer.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f wine_dma_infer
