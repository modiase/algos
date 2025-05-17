CC = gcc
CFLAGS = -Wall -Wextra -Werror -std=c11 -Wpedantic
TMP_DIR = .tmp.local
PROGRAM = $(TMP_DIR)/program

$(PROGRAM): main.c
	mkdir -p $(TMP_DIR) && $(CC) $(CFLAGS) -o $@ $<

clean:
	rm -rf $(TMP_DIR)

run: $(PROGRAM)
	./$(PROGRAM)