#ifndef CONFIG_H
#define CONFIG_H

typedef struct {
    const char *ds_path;
    const char *q_path;
    int h;
    int k;
    int x;
} Config;

int parse_args(int argc, char **argv, Config *cfg);

#endif
