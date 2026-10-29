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

---

## Staff note (added by course staff)

Course staff added this section when the project was imported as an example. The authors' text above is unchanged; where it and this note disagree, this note describes the repository as it is.

**Source and terms.** [Data for the 2018 Duolingo Shared Task on Second Language Acquisition Modeling (SLAM)](https://doi.org/10.7910/DVN/8SWHNO), Harvard Dataverse, doi:10.7910/DVN/8SWHNO. The dataset's terms of use on Dataverse read: "This work is released under the Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)." Use is allowed for non-commercial purposes with attribution; cite Settles et al. (2018) as above.

**How to get it.** Open the DOI page and choose Access Dataset. Dataverse asks for a short guestbook response before it releases the files, so the download cannot be scripted without one. The dataset has four files: `data_en_es.tar.gz` (46.9 MB), `data_es_en.tar.gz` (35.1 MB), `data_fr_en.tar.gz` (16.1 MB) and `starter_code.tar.gz`, the shared task's baseline. Each track archive holds that track's files. The code uses the train file, the dev file and the dev key; the PyMC notebooks name them `fr_en.slam.20190204.train`, `fr_en.slam.20190204.dev` and `fr_en.slam.20190204.dev.key`.

**Where the code expects the files.**

- The PyMC notebooks in `analysis/` use only the French track. Set `DATA_DIR` in the first configuration cell to the folder holding the three `fr_en.slam.20190204.*` files. The notebooks write their feature tables, posterior and predictions to a `processed/` folder inside it.
- The MAP scripts in `analysis/` expect the layout shown above under `slam/<track>/`, then a Parquet file per split made by `analysis/parse_slam.py`, for example `python analysis/parse_slam.py slam/es_en/dev slam/es_en_dev.parquet slam/es_en/dev.key`. As committed, the parser does not write the `morph` column that the other scripts read; the staff README explains this under "What does not run as committed".

**What is in this repository after all.** The line above that "no derived data artifacts are committed here" is not quite right. `demo_data.json` (652 KB), which the demo page reads, is derived from the dev sets: 300 tokens from the French course and 300 from the Spanish course, each with the token text, part of speech, exercise format, session type, the learner's country, days in the course, the number of earlier exposures, the true label, and the fitted model's predicted error probability, uncertainty values and 120 posterior draws. It has no learner IDs. Under CC BY-NC 4.0 it can stay in a non-commercial teaching repository with the citation above; it may not be reused commercially. Nothing else from the dataset is committed, and `.gitignore` keeps the raw files and the Parquet tables out.
