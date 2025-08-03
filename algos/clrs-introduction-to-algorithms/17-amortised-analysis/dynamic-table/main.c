#include <stdio.h>
#include <stdlib.h>

struct Table {
	int *data;
	int size;
	int capacity;
};

struct Table *Table_create(int capacity)
{
	struct Table *t = malloc(sizeof(struct Table));
	t->data = malloc(capacity * sizeof(int));
	t->size = 0;
	t->capacity = capacity;
	return t;
}

void Table_insert(struct Table *t, int value)
{
	if (t->size == t->capacity) {
		printf("Reallocating table from capacity %d to %d\n",
		       t->capacity, t->capacity * 2);
		t->data = realloc(t->data, t->capacity * sizeof(int));
		t->capacity *= 2;
	}
	t->data[t->size] = value;
	t->size++;
}

void Table_delete(struct Table *t, int index)
{
	if (index < 0 || index >= t->size) {
		return;
	}
	t->data[index] = t->data[t->size - 1];
	t->data[t->size - 1] = 0;
	t->size--;
	if (t->size < t->capacity / 4) {
		printf("Reallocating table from capacity %d to %d\n",
		       t->capacity, t->capacity / 2);
		t->data = realloc(t->data, t->capacity * sizeof(int));
		t->capacity /= 2;
	}
}

void print_table_size(struct Table *t)
{
	printf("Table size: %d\n", t->size);
}

void print_table_capacity(struct Table *t)
{
	printf("Table capacity: %d\n", t->capacity);
}

void print_table_data(struct Table *t)
{
	printf("[");
	for (int i = 0; i < t->size; i++) {
		printf("%d: %d", i, t->data[i]);
		if (i < t->size - 1) {
			printf(", ");
		}
	}
	printf("]\n");
}

int main()
{

	struct Table *t = Table_create(5);

	Table_insert(t, 0);
	Table_insert(t, 1);
	Table_insert(t, 2);
	Table_insert(t, 3);
	Table_insert(t, 4);

	print_table_data(t);
	print_table_size(t);
	print_table_capacity(t);

	Table_insert(t, 5);

	print_table_data(t);
	print_table_size(t);
	print_table_capacity(t);

	Table_delete(t, 0);
	Table_delete(t, 0);
	Table_delete(t, 0);
	Table_delete(t, 0);
	Table_delete(t, 0);
	Table_delete(t, 0);

	print_table_data(t);
	print_table_size(t);
	print_table_capacity(t);

	return 0;
}
