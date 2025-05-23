# Identity-By-Descent Analysis in HPRCv2

---

## Short term (almost instant) application

Using available vcfs from HPRCv2 and public data. It can be upgraded with network analysis.

### **Tracing HPRC haplotypes in public cohorts**
> *Where else does this haplotype appear?*

Search for IBD segments shared between HPRC haplotypes and individuals in public datasets like 1000 Genomes, HGDP or SGDP, and also can be extended to AllofUs. Helps estimate population frequency, geographic spread, or uniqueness of reference haplotypes.


### **Measuring effective contribution of each haplotype**
> *How much variation does this sample really add?*

Use IBD to assess how genealogically redundant each haplotype is relative to the panel. It can be used for estimating unique haplotypic content. Can be integrated with diversity summaries from PCA/FST. Also, it can provide a framework for selecting new samples (which haplotype information is not covered) from different projects, such as All of Us.

---

## Medium-term application

Once the resource is started to be used, providing IBD as metadata can help to different tasks.

### **Validating structural variants via IBD sharing**
> *Who else carries this variant?*

Given an SV identified in an HPRC assembly, find individuals who share IBD at the relevant locus. Shared inheritance supports the variant's authenticity and relevance. Lack of IBD support may suggest a recent mutation or call uncertainty.


### **Support for structural imputation via IBD**
> *Can we fill in a missing region?*

If two assemblies share IBD over a complex region, one can inform the other for structural resolution. Facilitates inferred SV calling or gap filling in low-coverage or ambiguous regions.

---

## Development (medium long-term) project

Pangenome application can be used as a framework to re-evaluate IBD inference.

### **Exploring IBD estimation from the pangenome graph**
> *What does identity by descent mean in a graph context?*

Propose defining IBD as common trajectories through the pangenome graph, rather than intervals on a linear reference. Enables novel metrics of haplotype conservation, cohesion, or uniqueness across graph paths (for example, now IBD is considered one-to-one, without incorporating duplcations).
