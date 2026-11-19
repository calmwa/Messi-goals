# Approved bonus: context projection matrix

The dashboard includes a 3 by 3 matrix with three subspace groups:

- **Match context:** competition, venue, and opponent.
- **Goal context:** finishing type, playing position, and minute.
- **Career context:** club and season start year.

Each diagonal cell applies PCA to one group. Each off-diagonal cell applies PCA to the combined features from its row and column groups. Select a cell button to enlarge the plot. The existing goal-ID multiselect drives the same highlight in every matrix cell, the enlarged plot, and the main dashboard views. PCA is recomputed within each subspace, with standardized numeric values and one-hot categories; rare categories are grouped for safer opponent geometry.

This adapts the paper's Dimension Projection Matrix concept: matrix cells compare projections over selected dimension groups. It does not reproduce the full projection tree, automated subspace navigation, or all interactions described by Yuan, Ren, Wang, and Guo (2013). [IEEE VIS paper page](https://ieeevis.org/year/2013/paper/infovis/dimension-projection-matrixtree-interactive-subspace-visual-exploration-and-).
