#include "compare64.h"
#include <stdio.h>
#include <math.h>

void compare_results_f64(const MatrixF64 *queries,
                         const Neighbor64 *computed,
                         const MatrixI32 *ref_ids,
                         const MatrixF64 *ref_dist,
                         int k)
{
    size_t Q = queries->n;

    printf("\n==============================\n");
    printf(" CONFRONTO RISULTATI 64-BIT\n");
    printf("==============================\n\n");

    for (size_t qi = 0; qi < Q; qi++) {

        const Neighbor64 *nc        = &computed[qi * k];
        const int32_t *ref_ids_row  = &ref_ids->data[qi * k];
        const double *ref_dst_row   = &ref_dist->data[qi * k];

        printf("Query %zu:\n", qi);

        int ok_ids  = 1;
        int ok_dist = 1;

        for (int j = 0; j < k; j++) {

            int id_ok = (nc[j].id == ref_ids_row[j]);
            double diff = nc[j].dist_real - ref_dst_row[j];
            if (diff < 0.0) diff = -diff;
            int d_ok = (diff < 1e-9);   // precisione maggiore per double

            if (!id_ok) ok_ids = 0;
            if (!d_ok) ok_dist = 0;

            printf("  k=%d -> id: %d (ref %d)   dist: %.12lf (ref %.12lf)%s\n",
                   j,
                   nc[j].id, ref_ids_row[j],
                   nc[j].dist_real, ref_dst_row[j],
                   (id_ok && d_ok ? "  OK" : "  *** DIFFERENTE ***"));
        }

        if (ok_ids && ok_dist)
            printf("  Tutto coincide.\n\n");
        else
            printf("  Differenze rilevate.\n\n");
    }
}
