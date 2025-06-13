#include <stdlib.h>
#include <string.h>
#include <stddef.h>

#define _DEFAULT_SIZE 64
#define MAX_LOAD_FACTOR 0.75
#define MIN_LOAD_FACTOR 0.25
#define GROWTH_FACTOR 2

enum ring_buffer_result {
    RING_BUFFER_SUCCESS = 0,
    RING_BUFFER_FAILED = -1
};

struct ring_buffer {
    size_t head;
    size_t tail;
    size_t size; 
    size_t capacity;
    size_t element_size;
    void* data;
};

struct ring_buffer* ring_buffer_create(size_t element_size) {
    if (element_size == 0) return NULL;
    if (_DEFAULT_SIZE > SIZE_MAX / element_size) return NULL;
    struct ring_buffer* buffer = malloc(sizeof(struct ring_buffer));
    if (!buffer) return NULL;

    buffer->head = 0;
    buffer->tail = 0;
    buffer->capacity = _DEFAULT_SIZE;
    buffer->element_size = element_size;
    buffer->data = malloc(_DEFAULT_SIZE * element_size);
    if (!buffer->data) {
        free(buffer);
        return NULL;
    }
    buffer->size = 0;
    return buffer;
}

static void _ring_buffer_resize(struct ring_buffer* buffer) {
    if (buffer->size == 0) {
        return;
    }
    float load_factor = (float)buffer->size / (float)buffer->capacity;
    if (load_factor <= MAX_LOAD_FACTOR && load_factor >= MIN_LOAD_FACTOR) {
        return;
    }

    size_t new_capacity = 0;
    if (load_factor > MAX_LOAD_FACTOR) {
        new_capacity = buffer->capacity * GROWTH_FACTOR;
    } else if(load_factor < MIN_LOAD_FACTOR) {
        new_capacity = buffer->capacity / GROWTH_FACTOR;
        if (new_capacity < 1) new_capacity = 1;
    }

    void* new_data = malloc(new_capacity * buffer->element_size);
    if (!new_data) {
        return;
    }
    for (size_t o = 0; o < buffer->size; o++) {
        memcpy((char*)new_data + o * buffer->element_size, (char*)buffer->data + ((buffer->head + o) % buffer->capacity) * buffer->element_size, buffer->element_size);
    }
    free(buffer->data);
    buffer->data = new_data;
    buffer->capacity = new_capacity;
    buffer->head = 0;
    buffer->tail = buffer->size;
}

void ring_buffer_destroy(struct ring_buffer* buffer) {
    if (buffer) {
        free(buffer->data);
        free(buffer);
    }
}

static int _ring_buffer_enqueue_internal(struct ring_buffer* buffer, const void* value) {
    if (buffer->size == buffer->capacity) {
        return RING_BUFFER_FAILED;
    }
    memcpy((char*)buffer->data + buffer->tail * buffer->element_size, value, buffer->element_size);
    buffer->tail = (buffer->tail + 1) % buffer->capacity;
    buffer->size++;
    return RING_BUFFER_SUCCESS;
}

static int _ring_buffer_dequeue_internal(struct ring_buffer* buffer, void* value) {
    if (buffer->size == 0) {
        return RING_BUFFER_FAILED;
    }
    memcpy(value, (char*)buffer->data + buffer->head * buffer->element_size, buffer->element_size);
    buffer->head = (buffer->head + 1) % buffer->capacity;
    buffer->size--;
    return RING_BUFFER_SUCCESS;
}

int ring_buffer_enqueue(struct ring_buffer* buffer, const void* value) {
    if (!buffer || !value) return RING_BUFFER_FAILED;
    int result = _ring_buffer_enqueue_internal(buffer, value);
    if (result == RING_BUFFER_SUCCESS) {
        _ring_buffer_resize(buffer);
    }
    return result;
}

int ring_buffer_dequeue(struct ring_buffer* buffer, void* value) {
    if (!buffer || !value) return RING_BUFFER_FAILED;
    int result = _ring_buffer_dequeue_internal(buffer, value);
    if (result == RING_BUFFER_SUCCESS) {
        _ring_buffer_resize(buffer);
    }
    return result;
}

size_t ring_buffer_size(const struct ring_buffer* buffer) {
    if (!buffer) return 0;
    return buffer->size;
}

size_t ring_buffer_capacity(const struct ring_buffer* buffer) {
    if (!buffer) return 0;
    return buffer->capacity;
}

int ring_buffer_is_empty(const struct ring_buffer* buffer) {
    if (!buffer) return 1;
    return buffer->size == 0;
}

int ring_buffer_is_full(const struct ring_buffer* buffer) {
    if (!buffer) return 0;
    return buffer->size == buffer->capacity;
}
