#include <stdio.h>
#include "audio_preprocessor_int8_model_data.h"

int main() {
 
    // Specify the filename for the binary file.
    const char* filename = "preprocessor.tflite";

    // Open the file for writing in binary mode.
    FILE* file = fopen(filename, "wb");
    if (file == NULL) {
        perror("Error opening file");
        return 1;
    }

    // Write the entire array to the file.
    size_t numElements = sizeof(g_audio_preprocessor_int8_tflite) / sizeof(g_audio_preprocessor_int8_tflite[0]);
    fwrite(g_audio_preprocessor_int8_tflite, sizeof(int), numElements, file);

    // Close the file.
    fclose(file);

    printf("Array successfully written to %s\n", filename);

    return 0;
}
