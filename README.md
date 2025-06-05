# Identity-By-Descent Analysis in HPRCv2

Using available vcfs from HPRCv2 and public data: https://data.humanpangenome.org/raw-sequencing-data

### **Tracing HPRC haplotypes in public cohorts**
> *Where else does this haplotype appear?*

Search for IBD segments shared between HPRC haplotypes and individuals in public datasets like 1000 Genomes, HGDP or SGDP, and also can be extended to AllofUs. It can help to estimate population frequency, geographic spread, or uniqueness of reference haplotypes.


### **Measuring effective contribution of each haplotype**
> *How much variation does this sample really add?*

Use IBD to assess how genealogically redundant each haplotype is relative to the panel. It can be used for estimating unique haplotypic content. Can be integrated with diversity summaries from PCA/FST. Also, it can provide a framework for selecting new samples (which haplotype information is not covered) from different projects, such as All of Us.

---

Once the resource is started to be used, providing IBD as metadata can help to different tasks.

### **Validating structural variants via IBD sharing**
> *Who else carries this variant?*

Given an SV or other variant of interest identified in an HPRC sample, find individuals who share IBD at the relevant locus. Shared inheritance supports the variant's authenticity and relevance. Lack of IBD support may suggest a recent mutation or call uncertainty.

---

## Dev

Pangenome application can be used as a framework to re-evaluate IBD inference.

### **Exploring IBD estimation from the pangenome graph**
> *What does identity by descent mean in a graph context?*

Propose defining IBD as common trajectories through the pangenome graph, rather than intervals on a linear reference. 
