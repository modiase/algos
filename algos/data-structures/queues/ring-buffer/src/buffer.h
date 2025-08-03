#ifndef RING_BUFFER_H
#define RING_BUFFER_H

#include <stdlib.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

struct ring_buffer;

enum ring_buffer_result {
    RING_BUFFER_SUCCESS = 0,
    RING_BUFFER_FAILED = -1
};

struct ring_buffer *ring_buffer_create(size_t element_size);
void ring_buffer_destroy(struct ring_buffer *buffer);
int ring_buffer_enqueue(struct ring_buffer *buffer, const void *value);
int ring_buffer_dequeue(struct ring_buffer *buffer, void *value);

size_t ring_buffer_size(const struct ring_buffer *buffer);
size_t ring_buffer_capacity(const struct ring_buffer *buffer);
int ring_buffer_is_empty(const struct ring_buffer *buffer);
int ring_buffer_is_full(const struct ring_buffer *buffer);

#ifdef __cplusplus
}
#endif
#endif				/* RING_BUFFER_H */
