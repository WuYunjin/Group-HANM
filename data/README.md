

Our implementation of processing fMRI data can be found in this [jupyter notebook](real/fmri.ipynb) under 'real' folder.

The real data that is a subset of the enhanced NKI Rockland sample (http://fcon_1000.projects.nitrc.org/indi/enhanced/, Nooner et al, 2012)

Resting state fMRI scans (TR=645ms) of 102 subjects were preprocessed (https://github.com/fliem/nki_nilearn) and projected onto the Freesurfer fsaverage5 template (Dale et al, 1999, Fischl et al, 1999). For this example we use the time series of a single subject’s left hemisphere.

The Destrieux parcellation (Destrieux et al, 2010) in fsaverage5 space as distributed with Freesurfer is used to select different seed regions.

For our analysis we considered 7 regions of interest:
```
pcc(116)

left_acc (167) 

left_mtg (183)

left_ag (171)

right_acc (191) 

right_mtg (188) 

right_ag (242),

AG = Angular gyrus; MTG = Middle temporal gyrus;
PCC = Posterior cingulate cortex; ACC = Anterior cingulate cortex
```

​       For more about the dataset, please refer to this example: http://nilearn.github.io/auto_examples/01_plotting/plot_surf_stat_map.html

