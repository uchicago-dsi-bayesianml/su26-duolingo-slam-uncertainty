# Data — Duolingo SLAM 2018 (not distributed in this repository)

The dataset is **© 2016 Duolingo, Inc.**, released under the
**Creative Commons Attribution-NonCommercial 4.0 International license
(CC BY-NC 4.0)**. We do not redistribute it here.

## Download

Get the three track archives from the Harvard Dataverse:

> **https://doi.org/10.7910/DVN/8SWHNO**
> (`data_es_en.tar.gz`, `data_fr_en.tar.gz`, `data_en_es.tar.gz`)

Extract so the analysis scripts find:

```
slam/
├── es_en/  train  dev  dev.key
├── fr_en/  train  dev  dev.key
└── en_es/  train  dev  dev.key
```

Then run `python analysis/parse_slam.py slam/es_en/train slam/es_en_train.parquet`
(and similarly for each file; dev files take the `.key` path as a third argument).

## Required citation

Per the dataset's CITING.md, any work using the data **or its derivatives** must
cite:

> B. Settles, C. Brust, E. Gustafson, M. Hagiwara, and N. Madnani. 2018.
> Second Language Acquisition Modeling. In *Proceedings of the NAACL-HLT
> Workshop on Innovative Use of NLP for Building Educational Applications
> (BEA)*. ACL.

```bibtex
@inproceedings{slam18,
  author = {B. Settles and C. Brust and E. Gustafson and M. Hagiwara and N. Madnani},
  booktitle = {Proceedings of the NAACL-HLT Workshop on Innovative Use of
               NLP for Building Educational Applications (BEA)},
  publisher = {ACL},
  title = {Second Language Acquisition Modeling},
  year = {2018},
  doi = {10.7910/DVN/8SWHNO},
  url = {https://doi.org/10.7910/DVN/8SWHNO}
}
```

Note the **NonCommercial** restriction: it governs use of the data and
derivatives (parquet conversions, fitted posterior draws over the data, etc.),
which is one reason no derived data artifacts are committed here either.
