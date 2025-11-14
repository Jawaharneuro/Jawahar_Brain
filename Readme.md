\# Automated Volumetric Analysis of Fetal Spina Bifida Atlas (Jawahar\_Brain)



This repository contains Python code and results for automated volumetric analysis

of the \*\*Spina Bifida Aperta fetal brain MRI atlas\*\* developed by King’s College London.



Using publicly available super-resolution fetal brain MRI data and parcellation masks,

the code computes:



\- Total brain volume (voxels) for each gestational age (GA 21–34 weeks)

\- Regional voxel counts from the atlas parcellation

\- Brain growth trends with gestational age

\- Correlation and basic statistics

\- Distribution of brain volumes and outlier detection



The project accompanies the manuscript:



> \*\*“Automated Quantitative Analysis of Fetal Brain Development in Spina Bifida Aperta Using a Spatio-Temporal MRI Atlas” – Jawahar Sri Prakash Thiyagarajan\*\*



---



\## Data



The data are \*not\* stored in this repository.



Fetal Spina Bifida Aperta atlas:



\- King’s College London / Fidon et al. (2022)

\- Synapse project: https://www.synapse.org/#!Synapse:syn25887675/wiki/611424



Download the atlas and set the local `data\_dir` path in `code/atlas\_volume\_analysis.py`.



Example folder layout expected by the script:



```text

SpinaBifidaAtlas\_v2/

&nbsp;   fetal\_SB\_atlas\_GA21\_notoperated/

&nbsp;       srr.nii.gz

&nbsp;       mask.nii.gz

&nbsp;       parcellation.nii.gz

&nbsp;       lmks.nii.gz

&nbsp;   fetal\_SB\_atlas\_GA22\_notoperated/

&nbsp;   ...

&nbsp;   fetal\_SB\_atlas\_GA34\_operated/



