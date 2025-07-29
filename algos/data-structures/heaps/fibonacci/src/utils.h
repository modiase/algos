#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <stdlib.h>

/**
 * CHECK_ALLOC_FATAL - Check memory allocation and exit on failure
 * @ptr: pointer returned from malloc/calloc/realloc
 * 
 * If ptr is NULL, prints an error message to stderr and exits with code 1.
 * Usage: CHECK_ALLOC_FATAL(malloc(size));
 */
#define CHECK_ALLOC_FATAL(ptr) do { \
    if ((ptr) == NULL) { \
        fprintf(stderr, "Fatal: Memory allocation failed at %s:%d: ", \
                __FILE__, __LINE__); \
        perror(""); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

/**
 * CHECK_ALLOC - Check memory allocation and return error code
 * @ptr: pointer returned from malloc/calloc/realloc
 * @retval: value to return on allocation failure
 * 
 * If ptr is NULL, prints an error message and returns retval.
 * Usage: CHECK_ALLOC(ptr, -1);
 */
#define CHECK_ALLOC(ptr, retval) do { \
    if ((ptr) == NULL) { \
        fprintf(stderr, "Error: Memory allocation failed at %s:%d: ", \
                __FILE__, __LINE__); \
        perror(""); \
        return (retval); \
    } \
} while(0)

/**
 * SAFE_FREE - Safely free memory and set pointer to NULL
 * @ptr: pointer to free
 * 
 * Frees the memory and sets the pointer to NULL to prevent double-free bugs.
 * Usage: SAFE_FREE(my_ptr);
 */
#define SAFE_FREE(ptr) do { \
    if ((ptr) != NULL) { \
        free(ptr); \
        (ptr) = NULL; \
    } \
} while(0)

/**
 * ARRAY_SIZE - Get the number of elements in a static array
 * @arr: static array
 * 
 * Returns the number of elements in a statically allocated array.
 * Usage: int count = ARRAY_SIZE(my_array);
 */
#define ARRAY_SIZE(arr) (sizeof(arr) / sizeof((arr)[0]))

/**
 * MIN/MAX - Get minimum/maximum of two values
 * @a, @b: values to compare
 * 
 * Returns the smaller/larger of the two values.
 * Usage: int smaller = MIN(x, y);
 */
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

#endif /* UTILS_H */