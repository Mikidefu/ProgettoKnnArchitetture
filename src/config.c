#include "config.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int parse_args(int argc, char **argv, Config *cfg) {

    for (int i = 1; i < argc; ++i) {

        if (strcmp(argv[i], "-d") == 0 && i + 1 < argc)
            cfg->ds_path = argv[++i];

        else if (strcmp(argv[i], "-q") == 0 && i + 1 < argc)
            cfg->q_path = argv[++i];

        else if (strcmp(argv[i], "-h") == 0 && i + 1 < argc)
            cfg->h = atoi(argv[++i]);

        else if (strcmp(argv[i], "-k") == 0 && i + 1 < argc)
            cfg->k = atoi(argv[++i]);

        else if (strcmp(argv[i], "-x") == 0 && i + 1 < argc)
            cfg->x = atoi(argv[++i]);

        else {
            printf("Parametro non riconosciuto: %s\n", argv[i]);
            return -1;
        }
    }

    if (!cfg->ds_path || !cfg->q_path)
        return -1;

    return 0;
}
