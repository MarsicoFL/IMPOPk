# Identity-By-State and Identity-By-Descent Analysis in HPRCv2

Using available vcfs from HPRCv2 and public data: https://data.humanpangenome.org/raw-sequencing-data

### **Tracing HPRC haplotypes in public cohorts**
> *Where else does this haplotype appear?*

Search for IBD segments shared between HPRC haplotypes and individuals in public datasets like 1000 Genomes, HGDP or SGDP, and also can be extended to AllofUs. It can help to estimate population frequency, geographic spread, or uniqueness of reference haplotypes.


### **Measuring effective contribution of each haplotype**
> *How much variation does this sample really add?*

Use IBD to assess how genealogically redundant each haplotype is relative to the panel. It can be used for estimating unique haplotypic content. Can be integrated with diversity summaries from PCA/FST. Also, it can provide a framework for selecting new samples (which haplotype information is not covered) from different projects, such as All of Us.

---

## Dev

Pangenome application can be used as a framework to re-evaluate IBD inference.


### **Exploring IBS and IBD estimation from the pangenome graph**

Propose defining IBD as common trajectories through the pangenome graph, rather than intervals on a linear reference. 
We are using impg as a scaffold for IBS and IBD detection. This is part of impop.
