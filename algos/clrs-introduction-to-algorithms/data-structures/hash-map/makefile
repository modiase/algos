main: main.c hash-map.o
        gcc main.c hash-map.o -o heapmain -Wall -Wpedantic -Werror -std=c99

hash-map.o: hash-map.c hash-map.h
        gcc -c hash-map.c -o hash-map.o -Wall -Wpedantic -Werror -std=c99 -I./
