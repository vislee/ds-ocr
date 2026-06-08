/*
 * ds_platform_ocr.c - Non-Apple platform OCR stub.
 */

#include "ds_platform_ocr.h"

char *ds_platform_ocr_file(const char *image_path, int accurate, int verbosity) {
    (void)image_path;
    (void)accurate;
    (void)verbosity;
    return 0;
}
