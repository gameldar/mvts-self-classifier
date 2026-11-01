# Real accelerometer datasets to replace the synthetic generator

As of 2026-10-13.

`generate_sample_data.py` draws each behaviour as a fixed-frequency, fixed-amplitude sine
plus Gaussian noise with 1.0 g added on one chosen axis. Per-axis mean and std are
therefore sufficient statistics, k-means on six hand features reaches 1.000 mapping
accuracy, and no conclusion drawn on it transfers. This file lists real, labelled,
tri-axial accelerometer datasets that could replace it, with the conversion work each
one needs to reach `data/train.parquet` + `data/train_labels.parquet`.

File placed at the repo root as `datasets.md` because the repo keeps all prose notes as
root-level files (`README.md`, `accelerometer.md`) and has no `docs/` directory.

## Lead

**Best candidate: EweAcT** ([doi:10.57745/AM2AJI](https://doi.org/10.57745/AM2AJI)).
79 hours of 25 Hz tri-axial neck-collar data from 120 Romane ewes, per-sample behaviour
labels annotated from video in BORIS, already shipped as a single 62.6 MB Parquet file in
g with gravity present, under Etalab Open Licence 2.0. It is the same species, the same
sensor class and nearly the same recording geometry as the sheep paper this repo's model
was built against, and the loader is about 60 lines.

**Best immediately-available stopgap: UCI HAR "Inertial Signals"**
([UCI 240](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)).
`total_acc_{x,y,z}` is already 10,299 windows of 128 samples per axis in g with gravity
included, so the conversion is a reshape and a column rename with no windowing decisions
at all. Off-domain, but it answers "does the model find structure in data that is not
synthetically separable" today.

**Priority 1 answer up front: the sheep paper's data is not published.** Turner et al.
2023 (Elsevier PII S2214317322000415) has no data-availability statement, no supplementary
material and no repository deposit. Contact details and the citation to quote are in the
first section below.

## 1. The sheep CNN-LSTM paper (PII S2214317322000415)

**Superseded 2026-10-13: we have this data.** The raw recordings are in the repo at
`data/sheep-behaviour` - 29 CSVs, 82,254 segments of 10 s, both the halter (30 Hz) and ear
(25 Hz) accelerometers, with an observed `class` per segment. This repo's author is an
author of the paper, so the availability question below is moot here; `sheep_data.py`
converts it. The section is kept for the citation and for anyone else who needs to ask.


### Identification

Turner, K. E., Thompson, A., Harris, I., Ferguson, M., & Sohel, F. (2022). Deep learning
based classification of sheep behaviour from accelerometer data with imbalance.
*Information Processing in Agriculture*, 10(3), 377-390.
[doi:10.1016/j.inpa.2022.04.001](https://doi.org/10.1016/j.inpa.2022.04.001)

Note the year discrepancy on the record itself: the Murdoch repository cover page cites it
as 2022, the running head on the article pages reads "Information Processing in Agriculture
10 (2023) 377-390", and it was received 2 September 2021, accepted 12 April 2022, available
online 18 April 2022. Both 2022 and 2023 appear on primary sources.

### Access

Open access, **not paywalled**. CC BY-NC-ND 4.0, stated twice in the article front matter.
`sciencedirect.com` returned HTTP 403 to automated fetches, but the full accepted PDF is
served openly from the
[Murdoch University research portal](https://researchportal.murdoch.edu.au/view/pdfCoverPage?instCode=61MUN_INST&filePid=13152686830007891&download=true)
(15 pages) and the publisher's own
[KeAi page](https://www.aeeisp.com/ipa/article/doi/10.1016/j.inpa.2022.04.001) resolves.
I read the complete 15-page PDF.

### Is the data public? No.

Searching the full text of the article for "data availability", "repository", "Mendeley",
"Dryad", "Zenodo", "figshare", "GitHub", "supplementary" and "on request" returns **no
matches at all**. The only hits for "licence" are the two CC BY-NC-ND statements. The
article ends with CRediT, Declaration of Competing Interest, Acknowledgements and
References, with no data or code statement between them. There is no supplementary
material list. I found no deposit on Mendeley Data, Dryad, Zenodo, figshare, PANGAEA,
Movebank or GitHub, and no dataset DOI in DataCite under these authors.

The same group's follow-up paper, Turner et al., "Lambing event detection using deep
learning from accelerometer data", *Computers and Electronics in Agriculture* (2023),
[doi:10.1016/j.compag.2023.107787](https://doi.org/10.1016/j.compag.2023.107787), carries
the Elsevier boilerplate "Data will be made available on request" - which is the strongest
available signal that the grazing data is likewise request-only. I did not open that
article's full text; this is from a search result, so treat it as unconfirmed.

### Who to contact

- **Ferdous Sohel** (corresponding author), Discipline of Information Technology, Murdoch
  University, 90 South Street, Murdoch, WA 6150, Australia. `F.Sohel@murdoch.edu.au`
  (printed in the article's corresponding-author footnote).
- **Kirk E. Turner** (first author, Data curation role in CRediT), same discipline.
- Data curation is also credited to Andrew Thompson (Murdoch), Ian Harris and Mark
  Ferguson (neXtgen Agri, Christchurch NZ).

Quote this citation when asking:

> Turner, K. E., Thompson, A., Harris, I., Ferguson, M., & Sohel, F. Deep learning based
> classification of sheep behaviour from accelerometer data with imbalance. Information
> Processing in Agriculture 10(3), 377-390. https://doi.org/10.1016/j.inpa.2022.04.001

Ask specifically for the per-sheep CSVs described in section 2.3 ("two for each sheep; one
containing the ActiGraph jaw mounted data, and the other containing the Axivity ear tag
data"), not the SMOTE-augmented or 5-fold CSVs, which are derived.

### What the data is, if they send it

| Field | Value |
|---|---|
| Sensors | ActiGraph (Pensacola FL) jaw/halter mounted, **30 Hz**; Axivity (Newcastle UK) ear mounted, **25 Hz** |
| Axes / units / range | Tri-axial; units, dynamic range and gravity handling **not stated in the article** |
| Placement and orientation | Jaw-mounted under the chin via halter, and ear tag. Fig. 2 is a schematic of both; no numeric axis convention given |
| Label granularity | One label per **10 s** block |
| How labelled | Video recordings during daylight, subdivided into 10 s blocks, behaviours allocated by human observation. Steps per 10 s window also counted from video |
| Animals | 30 ewes total; 9 instrumented in Muresk Dry Pasture (Dec 2018), 10 in Muresk Stubble (Feb 2019), disjoint sets |
| Classes | sitting, standing, walking, grazing, ruminating, combined into 13 compound classes |
| Raw per-sample data | **Yes** - 900 ActiGraph points + 750 Axivity points per 10 s window, i.e. 3 axes x 300 and 3 axes x 250, concatenated to a 1650-column array |
| Class counts (Muresk Dry Pasture / Muresk Stubble) | sitting 9757/2469; standing_grazing 6652/5253; standing 6295/6392; standing_ruminating 2088/1684; sitting_ruminating 1498/836; walking 907/1715; standing_walking_grazing 888/5253; walking_grazing 653/218; standing_walking 380/1470; standing_walking_ruminating 39/89; walking_ruminating 14/14; sitting_walking 4/1; standing_walking_grazing_ruminating 3/3; walking_grazing_ruminating 1/1 |
| Licence | Unknown. The article is CC BY-NC-ND 4.0; that says nothing about the data |

Conversion if obtained: pick one sensor (Axivity at 25 Hz gives 250 samples/axis per 10 s
block; ActiGraph 300). W=250 means `--data-size 750`; W=300 means `--data-size 900`. The
10 s block is already the window, so no boundary problem. Drop the 1650-column merged form
and take one sensor's 3 x W slice, reordered to blocked x/y/z. Keep the sheep identifier
for a leave-one-sheep-out split. Roughly 40 lines.

## 2. Published grazing-livestock sets with observed labels

### 2.1 EweAcT - sheep, neck collar, 25 Hz (recommended)

| Field | Value |
|---|---|
| Name / citation | Paper: Riaboff, L., Andriamampandry, N. A., Bompa, J.-F., Aletru, M., Durand, C., Douls, S., et al. *EweAcT: Ewe behaviour aligned to accelerometer data for activity monitoring in extensive grazing systems*. [arXiv:2608.09943](https://arxiv.org/abs/2608.09943), submitted 1 Jul 2026, [doi:10.48550/arXiv.2608.09943](https://doi.org/10.48550/arXiv.2608.09943), arXiv nonexclusive-distrib 1.0. Dataset (titled *...aligned to accelerometer time-series...*, per the repository record): [doi:10.57745/AM2AJI](https://doi.org/10.57745/AM2AJI), Recherche Data Gouv, published 2026-06-26, version 1.1 |
| Licence | **Etalab Open Licence 2.0** ([SPDX etalab-2.0](https://spdx.org/licenses/etalab-2.0.html)), from the repository API. Permits reproduction, redistribution, adaptation and commercial use, requiring attribution of source and date. Derivative works allowed |
| Landing page | <https://entrepot.recherche.data.gouv.fr/dataset.xhtml?persistentId=doi:10.57745/AM2AJI> - HTTP 200. Note the HTML landing page is behind an Anubis bot check for automated clients; the Dataverse API at `/api/datasets/:persistentId/?persistentId=doi:10.57745/AM2AJI` returns 200 plainly |
| Direct download (main file) | <https://entrepot.recherche.data.gouv.fr/api/access/datafile/749162> - verified, HTTP 303 to a signed S3 URL |
| Sample rate | 25 Hz, fixed (AX3 configured at 25 Hz for both batches) |
| Axes / units / range | 3 axes. **Units are g, gravity present** - verified by range-reading the first 1.5 KB of the repository's own AX3 sample file (`93942_34057_2024-06-11`, datafile 749157): rows read `2024-06-11 08:04:53.439,-1.039063,-0.078125,-0.031250` at 40 ms spacing, i.e. about -1 g resting on X with a 1/128 g quantisation step. Sensitivity configured at **±4 g**. The data article does not name the unit in words; this is inferred from the sample file |
| Gravity | Included. "No normalization was applied" |
| Label granularity | **Per sample** - one `Behaviour` string on every one of the 7,108,285 rows |
| How labelled | Batch 1: four fixed CCTV cameras 4 m above 12 x 12 m paddocks, annotated per focal animal in BORIS by two trained observers, inter-observer Cohen's kappa > 0.80 at 0.5 s intervals. Clock-aligned to the camera UTC clock by a rotating-cylinder pattern injected into the accelerometer signal, with linear drift correction; residual offsets 0-3.2 s (mean 0.42 s) corrected by expert inspection. Batch 2 (Moving): start/stop times of 10-minute track walks, cross-checked against chest-mounted GoPro video and audio |
| Classes and imbalance | Of 7,108,285 readings: Ruminating 32.5% (63 ewes), Grazing 29.0% (68), Moving 18.2% (118), Resting 17.1% (66), Other 3.3% (68). Read off Figure 1 of the data article; the article gives percentages, not absolute counts |
| Placement and orientation | Neck collar, accelerometer in a locker on the right side of the dorsal part. **X = head-to-tail, Y = lateral, Z = top-to-bottom.** A 100 g ventral counterweight prevents collar rotation, so orientation is consistent between animals. This is the best-documented orientation of any candidate here |
| Raw per-sample data | **Yes.** No features, no summaries |
| Download size | `EweAcT.parquet` 65,645,405 bytes (62.6 MB). Whole deposit 110.4 MB across 7 files (dataset, metadata XLSX/TAB, the alignment Jupyter notebook, and three sample files) |
| Animals | 120 Romane ewes, born 2021-2024, one breed, females only, INRAE La Fage, Aveyron, France |

Known caveats, from the article's own Limitations section: one breed, mostly ewe lambs;
much of Resting and Ruminating was recorded above 20 °C with heat-related behaviour; and
the Moving class comes almost entirely from batch 2 (November 2025, >13 h from 82 ewes on
tracks), a different session from the rest, so session effects are confounded with that
class.

**Conversion work.** Minimal.

- `EweAcT.parquet` has `AnimalID, DateTime, AccX, AccY, AccZ, Behaviour, Modifier_1..3,
  Comment`. Keep the first six.
- Unannotated stretches were removed, so timestamps are not continuous. Segment on a
  change of `AnimalID`, a change of `Behaviour`, or a `DateTime` gap above about 0.06 s
  (1.5 sample periods at 25 Hz).
- Window **within** segments only: W=150 samples is 6.0 s at 25 Hz, matching the synthetic
  default and keeping `--data-size 450`. Non-overlapping is the honest default; 50%
  overlap doubles window counts but correlates train and eval windows.
- **No window ever spans a label boundary**, because segments are by construction
  single-animal and single-behaviour. Drop the tail remainder of each segment.
- No unit conversion. Cast to float32 and emit blocked `x_0..x_149, y_0..y_149,
  z_0..z_149`. `Timestamp` = first sample's `DateTime` of the window.
- Expected windows at W=150 non-overlapping, before segment-remainder losses: Ruminating
  about 15,400, Grazing about 13,700, Moving about 8,600, Resting about 8,100, Other about
  1,560. The rare class clears "several hundred" by a factor of about 3 even after losses.
- Subjects: 120 `AnimalID`s. Hold out whole animals for evaluation; the metadata table
  gives genetic line, month, slope and thermal status per animal-period if you want
  stratified splits. The dataset has no train/test split of its own.
- About 60 lines of loader.

### 2.2 CabriTrack - Creole goats, horn mounted, 25 Hz

| Field | Value |
|---|---|
| Name / citation | Faillot, L., Troupe, W., Bonneau, M. (2025). CabriTrack: Accelerometer data for automated behavioural monitoring of grazing Creole goats. *Data in Brief* 59, 111431. [doi:10.1016/j.dib.2025.111431](https://doi.org/10.1016/j.dib.2025.111431). Dataset: [doi:10.57745/CYF9QW](https://doi.org/10.57745/CYF9QW), Recherche Data Gouv, published 2024-07-17 |
| Licence | **Conflict.** Crossref records the article as CC BY 4.0 (version of record). The repository API records the **dataset** licence as Etalab Open Licence 2.0, while the article text as rendered on PMC says CC BY. Both permit redistribution and derivatives with attribution, so the practical difference is small, but the two sources disagree on which licence governs the data |
| Direct download | <https://entrepot.recherche.data.gouv.fr/api/access/datafile/285154> - verified, HTTP 303 |
| Sample rate | 25 Hz |
| Axes / units / range | 3 axes, Axivity AX3. Units g (same device family and export path as EweAcT); dynamic range **not stated** in the repository description |
| Gravity | Included (raw AX3 output, no filtering described) |
| Label granularity | **Per sample** - `Behaviour` and `Behaviour_num` on each of 12,756,669 rows, plus `Sequence_num` grouping contiguous same-behaviour runs |
| How labelled | CCTV video of 2-5 animals at a time in an experimental field, annotated manually in BORIS, UTC-aligned |
| Classes and imbalance | Grazing 67.7 h, Resting 59.5 h, Ruminating/Chewing 10.6 h, Other 6.2 h, Displacement 0.6 h. These per-class hours come from the PMC rendering of the Data in Brief article, which I opened once successfully and could not re-open afterwards (captcha); note the hours sum to about 144.6 h while 12,756,669 rows at 25 Hz is 141.7 h, so the two figures do not reconcile exactly |
| Placement and orientation | **Inside of the left horn**, taped. X along the horn, Y along the back of the animal, Z perpendicular in the same plane as Y. Documented, but a horn is a very different lever arm from a collar or jaw mount |
| Raw per-sample data | **Yes.** `Time, Animal_id, Behaviour, Behaviour_num, X, Y, Z, Sequence_num` |
| Download size | 867,745,990 bytes (867.7 MB) in a single plain-text file, `Final-DataPaper-Horn.txt` |
| Animals | 59 Creole goats, Guadeloupe (INRAE) |

**Conversion work.** Same shape as EweAcT, with two differences: `Sequence_num` already
does the segmentation for you, and the 867 MB text file must be read in chunks (pandas
`chunksize`, or convert once to Parquet). W=150 is again 6.0 s. Expected windows,
non-overlapping: Grazing about 40,600, Resting about 35,700, Ruminating about 6,360, Other
about 3,720, **Displacement about 360**. Displacement only just clears "several hundred",
so it is the class that decides whether this dataset is scoreable for you. About 50 lines
plus a chunked reader.

### 2.3 Goats and sheep collar tags, 200 Hz (Kamminga et al.)

| Field | Value |
|---|---|
| Name / citation | Kamminga, J. W., Havinga, P. J. M., Meratnia, N. (2017). *Generic online animal activity recognition on collar tags*. Dataset, DANS Data Station Life Sciences. [doi:10.17026/dans-zp6-fmna](https://doi.org/10.17026/dans-zp6-fmna). Companion paper: UbiComp/ISWC '17, [doi:10.1145/3123024.3124407](https://doi.org/10.1145/3123024.3124407) |
| Licence | **CC0-1.0** (public domain dedication), from the DANS Dataverse API. The most permissive licence of any candidate here |
| Landing page | <https://lifesciences.datastations.nl/dataset.xhtml?persistentId=doi:10.17026/dans-zp6-fmna> - verified, DOI resolves HTTP 200 |
| Sample rate | 200 Hz, all sensors |
| Axes / units / range | Accelerometer, gyroscope, magnetometer, temperature, pressure. Units and dynamic range **not stated** in the repository description; the Readme.pdf inside the archive is said to have more |
| Gravity | Not stated |
| Label granularity | Not stated in the repository description. The companion papers describe labelling from video via a MATLAB GUI |
| How labelled | Video recordings used as reference, labelled through a MATLAB GUI application |
| Classes | 9 activities. The literature lists lying, standing, grazing, fighting, shaking, scratching-biting, walking, trotting, running. Per-class counts **not stated** on the repository record |
| Placement and orientation | Collar around the neck - **"the orientation was not fixed (the collars were prone to rotation around the neck)"**. This is a direct disqualifier against the orientation requirement, and the group's own follow-up work is specifically about orientation-independent features |
| Raw per-sample data | Yes - per-animal CSV and MATLAB `.mat` files (G1-G4 goats, S1-S2 sheep), roughly 240-270 MB per CSV |
| Download size | 2,243.8 MB across 14 files |
| Animals | 4 goats, 2 sheep, one day each |

Six animals and unfixed collar orientation make this a poor fit for a model that leans on
the orientation cue, despite the excellent licence. Related deposits from the same group,
both CC0: *Multi Sensor-Orientation Movement Data of Goats*
([doi:10.17026/dans-xhn-bsfb](https://doi.org/10.17026/dans-xhn-bsfb), 100 Hz, 5 goats,
six simultaneously-worn sensor nodes at different orientations) and *Horsing Around*
([doi:10.4121/uuid:2e08745c-4178-4183-8551-f248c992cb14](https://doi.org/10.4121/uuid:2e08745c-4178-4183-8551-f248c992cb14),
100 Hz, 18 horses and ponies, 1.2 M 2-second samples of which 93,303 labelled from video,
17 activity categories, published in *Data* 4(4):131,
[doi:10.3390/data4040131](https://doi.org/10.3390/data4040131)).

### 2.4 Free-grazing cattle IoT collar - rejected

Morales-Vargas, D., et al. (2025). A dataset for detecting walking, grazing, and resting
behaviors in free-grazing cattle using IoT collar IMU signals. *Frontiers in Veterinary
Science* 12:1630083.
[doi:10.3389/fvets.2025.1630083](https://doi.org/10.3389/fvets.2025.1630083). Data on
GitHub at <https://github.com/WASP-lab/db-cow-walking>. Neck collar, 10 dairy cows, labels
by manual annotation of PTZ camera video, raw CSV time series (not features only), units
m/s². **Rejected on size and rate**: 10 Hz, 7 h 34 min total, with Walking at 53 min
(about 31,800 samples) and Resting at 10 events. At 10 Hz a 6 s window is 60 samples and
Walking yields roughly 530 windows while Resting yields roughly 1,900 from only 10 events,
so windows within a class are near-duplicates.

### 2.5 CSIRO cattle (Arm18 / Arm20c / Arm20e) - not located as a public deposit

Arablouei, Bishop-Hurley and colleagues at CSIRO describe labelled collar (50 Hz,
InvenSense MPU-9250) and ear-tag (62.5 Hz, Bosch BMA400) cattle datasets annotated from
GoPro video, across several papers
(e.g. [arXiv:2111.12295](https://arxiv.org/abs/2111.12295),
[arXiv:2206.12078](https://arxiv.org/abs/2206.12078)). I could not find a public
repository deposit for Arm18, Arm20c or Arm20e in DataCite or on the CSIRO Data Access
Portal. Treat as request-only until proven otherwise; I did not open the full text of
these preprints, so the sensor figures above are from search results, not verified.

## 3. Human activity recognition stopgaps

### 3.1 UCI HAR - the 561-feature vs Inertial Signals distinction

This distinction is the single most important thing on this page for anyone reaching for
UCI HAR.

- `train/X_train.txt` and `test/X_test.txt` hold the **561-feature vectors** (mean, std,
  energy, FFT coefficients and so on per window), normalised and bounded to [-1, 1].
  **These are unusable for us**: they are summary features, not time series. They are also
  the files most tutorials load.
- `train/Inertial Signals/` and `test/Inertial Signals/` hold the **raw windows**: nine
  files of 128 values per row - `total_acc_{x,y,z}`, `body_acc_{x,y,z}`,
  `body_gyro_{x,y,z}`. These are what we want.

I verified the file list by opening the published zip's directory listing (both
`Inertial Signals` folders present with all nine files each, `total_acc_x_train.txt` at
15,071,600 bytes, `total_acc_x_test.txt` at 6,041,350 bytes).

**Gravity split.** The README states verbatim: "The acceleration signal from the
smartphone accelerometer X axis in **standard gravity units 'g'**. Every row shows a 128
element vector" for `total_acc_*`, and "The body acceleration signal obtained by
**subtracting the gravity** from the total acceleration" for `body_acc_*`. The separation
used a Butterworth low-pass filter at 0.3 Hz cutoff.

This matters for us in both directions. `total_acc` is the direct analogue of our
generator's output (g, gravity present), so it needs no conversion - but it also means
sitting, standing and laying are separable from the gravity vector alone, which is exactly
the orientation shortcut the synthetic data hands the model. `body_acc` removes that
shortcut and is the harder, more informative test. Running both is cheap and the
difference between the two scores is itself the result.

| Field | Value |
|---|---|
| Citation | Reyes-Ortiz, J., Anguita, D., Ghio, A., Oneto, L., Parra, X. (2013). Human Activity Recognition Using Smartphones. UCI Machine Learning Repository. [doi:10.24432/C54S4K](https://doi.org/10.24432/C54S4K) |
| Licence | **CC BY 4.0** (stated on the UCI dataset page). Redistribution and derivatives allowed with attribution |
| Download | <https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip> - verified HTTP 200, 61,005,872 bytes (58.2 MB). The outer zip contains `UCI HAR Dataset.zip` plus a `.names` file |
| Sample rate | 50 Hz |
| Axes / units / range | 3 axes, g. Dynamic range of the Galaxy S II accelerometer **not stated** |
| Gravity | `total_acc` includes it; `body_acc` has it removed (0.3 Hz Butterworth) |
| Label granularity | **Per window** - one of six integers per 128-sample window, in `y_train.txt` / `y_test.txt` |
| How labelled | "The experiments have been video-recorded to label the data manually" |
| Windows per class (counted directly from `y_train.txt` + `y_test.txt`) | LAYING 1944 (1407 train / 537 test), STANDING 1906 (1374/532), SITTING 1777 (1286/491), WALKING 1722 (1226/496), WALKING_UPSTAIRS 1544 (1073/471), WALKING_DOWNSTAIRS 1406 (986/420). Total 10,299 (7352/2947) |
| Class imbalance | Near-balanced by design, roughly 1.4:1 between the largest and smallest class. **This is a real limitation**: it does not exercise the rare-class problem at all, which is the whole point of the original sheep paper |
| Placement and orientation | Samsung Galaxy S II on the waist. No numeric axis convention published |
| Raw per-sample data | Yes, as pre-cut windows (see above) |

**Conversion work.** Smallest of any candidate. Read the three `total_acc_*` files with
`numpy.loadtxt` (7352 x 128 for train, 2947 x 128 for test), `np.hstack` them in x, y, z
order - that is already the blocked layout - cast to float32, name the columns
`x_0..x_127, y_0..y_127, z_0..z_127`, and set `--data-size 384`. Map `y_*.txt` integers
through `activity_labels.txt` for the sidecar. There are no real timestamps in this
version, so synthesise a `pd.date_range` exactly as `generate_sample_data.py` does; nothing
downstream reads it. Windows already have 50% overlap baked in, which means adjacent rows
share 64 samples - for a train/eval split, split on `subject_train.txt` /
`subject_test.txt` (subjects 1-30, already disjoint between the published train and test
sets) rather than on rows. About 30 lines.

### 3.2 UCI HAPT (dataset 341) - UCI HAR with genuinely continuous raw signals

Reyes-Ortiz et al. (2015). Smartphone-Based Recognition of Human Activities and Postural
Transitions. UCI ML Repository.
[doi:10.24432/C54G7M](https://doi.org/10.24432/C54G7M). Same 30 subjects, same waist
placement, same 50 Hz, same g units, CC BY 4.0,
<https://archive.ics.uci.edu/dataset/341/smartphone+based+recognition+of+human+activities+and+postural+transitions>.

The difference that matters: `RawData/acc_expXX_userYY.txt` is the **unwindowed**,
continuous signal ("Every row is one acceleration sample (three axis) captured at a
frequency of 50Hz"), and `RawData/labels.txt` gives labels as **time ranges** -
`experiment, user, activity, start sample, end sample`. It adds six postural transition
classes (stand-to-sit, sit-to-stand, sit-to-lie, lie-to-sit, stand-to-lie, lie-to-stand)
which are genuinely rare, so the class imbalance is real rather than balanced by design.

Prefer this over dataset 240 if you want control over window length and overlap, or if you
want rare classes. Conversion: window within each labelled range only, discard the tail
remainder, and never let a window straddle two ranges. Note the raw file is the total
acceleration (gravity present); the body/gravity split exists only in the windowed branch.
About 50 lines.

### 3.3 PAMAP2

| Field | Value |
|---|---|
| Citation | Reiss, A. (2012). PAMAP2 Physical Activity Monitoring. UCI ML Repository. [doi:10.24432/C5NW2H](https://doi.org/10.24432/C5NW2H) |
| Licence | CC BY 4.0 (UCI page) |
| Download | <https://archive.ics.uci.edu/static/public/231/pamap2+physical+activity+monitoring.zip> - verified HTTP 200. 656.3 MB per the UCI page |
| Sample rate | 100 Hz (IMUs); heart rate about 9 Hz on a separate channel |
| Axes / units / range | 3 IMUs, each exposing **two** accelerometers: ±16 g and ±6 g, both 13-bit. **Units m/s²** |
| Gravity | Included (raw IMU output; no filtering described) |
| Label granularity | **Per sample** - column 2 of every row is the activity id |
| How labelled | Scripted protocol of 12 activities per subject plus optional activities, supervised |
| Classes | 18 activities. Per-class durations **not stated** on the UCI page |
| Placement | Wrist of dominant arm, chest, dominant-side ankle |
| Raw per-sample data | Yes. Space-separated `.dat`, 54 columns: timestamp, activity id, heart rate, then 17 columns per IMU |
| Subjects | 9 |

**Conversion work.** The most of any HAR candidate. Pick one IMU and one of its two
accelerometers (the ±16 g triplet is the conventional choice) - that is three of the 54
columns. **Divide by 9.80665 to get g.** Labels are per sample, so windows can straddle
boundaries: segment on runs of constant activity id per subject first, then window inside
runs, exactly as for EweAcT. Activity id 0 means "transient/other" and is conventionally
dropped. There are NaNs from dropped wireless packets that must be interpolated or whose
windows must be discarded. W=150 is only 1.5 s at 100 Hz; prefer W=300 (3.0 s,
`--data-size 900`) or downsample to 25 Hz to keep W=150 meaningful. About 80 lines.

### 3.4 WISDM v1.1 (Activity Prediction)

| Field | Value |
|---|---|
| Citation | Kwapisz, J. R., Weiss, G. M., Moore, S. A. (2010). Activity Recognition using Cell Phone Accelerometers. *Proc. Fourth International Workshop on Knowledge Discovery from Sensor Data* (at KDD-10), Washington DC. The lab asks that this paper be cited |
| Licence | **No licence stated.** The lab's own request is "when sharing or redistributing this dataset, we request that the readme.txt file is always included". Redistribution rights are therefore unclear; treat it as "use and cite, do not redistribute" |
| Download | <https://www.cis.fordham.edu/wisdm/includes/datasets/latest/WISDM_ar_latest.tar.gz> - verified HTTP 200, `content-length` 11,404,612 bytes (11.4 MB), last modified 2016-07-26 |
| Sample rate | 20 Hz. The paper: "we collected the accelerometer data every 50ms, so we had 20 samples per second" |
| Axes / units / range | 3 axes, **m/s²** (Android convention). The paper: gravity "causes the accelerometer to measure a value of 9.8 m/s² in the direction of the Earth's center" |
| Gravity | Included |
| Label granularity | Per sample, in the raw file `WISDM_ar_v1.1_raw.txt` (`user, activity, timestamp, x, y, z`) |
| How labelled | Controlled laboratory protocol: subjects were asked to perform each activity for set periods while a researcher-controlled app recorded; not video-annotated |
| Classes and imbalance (raw samples) | Walking 424,400 (38.6%), Jogging 342,177 (31.2%), Upstairs 122,869 (11.2%), Downstairs 100,427 (9.1%), Sitting 59,939 (5.5%), Standing 48,395 (4.4%). Total 1,098,207 |
| Placement | Android phone in the front trouser leg pocket. Paper's convention: y = up/down, z = forward leg movement, x = horizontal leg movement |
| Raw per-sample data | **Yes**, and a 46-attribute transformed ARFF (5,424 examples) sits alongside it. Load the `_raw.txt`, never the `.arff` |
| Subjects | The paper says 29 users; the v1.1 release masks user names as IDs 1-36. The two primary sources disagree; I did not open the readme to resolve it |

**Avoid WISDM v2 (Actitracker) for this purpose.** Its "unlabeled transformed" split of
1,369,349 examples carries labels that, in the lab's own words, are "the activity that our
system predicted the user to be performing" - model-inferred, which fails our requirement
outright. Only the 2,980,765-row labelled raw split is human-labelled, and even then "the
user physically specifies which activity is being performed".

**Conversion work.** Divide by 9.80665 for g. W=150 is 7.5 s at 20 Hz, keeping
`--data-size 450`. Segment per `(user, activity)` on timestamp continuity before
windowing. Expected windows at W=150 non-overlapping: Walking about 2,800, Jogging about
2,280, Upstairs about 819, Downstairs about 669, **Sitting about 399, Standing about 322**
- the two static classes only just clear "several hundred". The raw file is notoriously
malformed (trailing semicolons on each record, blank lines, zero and duplicated
timestamps); budget for cleanup. About 60 lines including the cleanup.

### 3.5 Other HAR candidates considered

- **CAPTURE-24**. Chan Chang, S., Walmsley, R., Gershuny, J., et al. *Capture-24: Activity
  tracker dataset for human activity recognition*, Oxford University Research Archive,
  [doi:10.5287/bodleian:NGx0JOMP5](https://doi.org/10.5287/bodleian:NGx0JOMP5), landing
  page <https://ora.ox.ac.uk/objects/uuid:99d7c092-d865-4a19-b096-cc16440cd001>. **CC BY.**
  Axivity AX3 on the wrist, **100 Hz**, ±8 g, 151 participants, about 4,000 hours of which
  more than 2,500 hours are labelled. Labels come from **Vicon Autograph wearable camera
  images (one photo every 20 s) plus sleep diaries**, annotated against the 2011
  Compendium of Physical Activity with over 200 unique labels, which are then mapped to
  coarser groupings. Raw per-sample CSV with text annotations; single `capture24.zip` of
  6.4 GB. The best free-living, genuinely imbalanced human set here, and the closest in
  device terms to EweAcT and CabriTrack (same AX3). Downsides: 6.4 GB, wrist placement
  with no fixed orientation, 20 s label resolution, and the units of the distributed x/y/z
  columns are **not stated** on either the ORA record or the GitHub README
  (<https://github.com/OxWearables/capture24>) - AX3 exports in g, but verify on download.
- **MotionSense** (<https://github.com/mmalekzadeh/motion-sense>). MIT licence, 50 Hz,
  iPhone 6s in the front trouser pocket, 24 subjects, 6 activities, raw per-sample CSV.
  **Two problems for us**: labels are self-started by the participant in the Crowdsense
  app rather than observed, and the distributed signals are CoreMotion's
  `userAcceleration` with `gravity` as separate columns, so gravity is already split out
  and must be re-added to match our format.
- **RealWorld (HAR) 2016**
  (<https://www.uni-mannheim.de/dws/research/projects/activity-recognition/dataset/dataset-realworld/>).
  15 subjects, 7 simultaneous body positions (chest, forearm, head, shin, thigh, upper
  arm, waist), 8 activities, about 10 min per activity per subject except jumping (about
  1.7 min), video-recorded sessions, roughly 3.5 GB. Sample rate, units and licence are
  **not stated** on the page I opened. The seven simultaneous positions make it the best
  candidate for testing placement sensitivity, if the missing details check out.
- **Opportunity** and **MHEALTH** (both on UCI) were not investigated; neither offers
  anything the sets above lack for our purpose.

## Recommended order of attack

1. **UCI HAR Inertial Signals, today.** 58 MB, 30 lines of loader, no windowing decisions,
   `--data-size 384`. Run the existing `sweep.py` on `total_acc` and again on `body_acc`.
   If mapping accuracy on `body_acc` is near the collapse floor (the share of the largest
   class, 0.189 here with six balanced classes) then the 1.000 on synthetic data was
   entirely the gravity-axis cue, which is the result worth knowing before anything else.
2. **EweAcT, this week.** 110 MB, already Parquet, 25 Hz, g with gravity, per-sample
   labels, documented fixed orientation, real 10:1 imbalance between Ruminating and Other,
   and a 120-animal subject split. This is the dataset the project should be benchmarking
   on. W=150 keeps `--data-size 450` unchanged, so nothing but the loader moves.
3. **Email Sohel and Turner in parallel with step 2.** The Turner data is the only set that
   matches the model's provenance exactly (jaw and ear mounts, 10 s blocks, published class
   counts), and the request costs one email. Expect a wait; do not block on it.
4. **CabriTrack as a cross-domain check** once EweAcT works. Same rate, same device family,
   same labelling software, different species and a very different mount - a clean test of
   whether whatever the model learns is about behaviour or about one collar geometry.
   867 MB, needs chunked reading, and Displacement at about 360 windows is the binding
   constraint.
5. **HAPT or PAMAP2** only if a per-sample-label, boundary-handling path is wanted on human
   data; both are strictly more work than EweAcT for less relevance.
6. **Skip** the Kamminga goat/sheep set (6 animals, unfixed collar orientation), the
   free-grazing cattle set (10 Hz, 7.5 h), and WISDM v2's unlabeled split (model-predicted
   labels).

## Provenance

Pages opened and read in full:

- Turner et al. accepted PDF via the Murdoch research portal (15 pages, complete text
  searched for data-availability wording)
- `aeeisp.com` KeAi article page for Turner et al.
- arXiv abs page and full 19-page PDF for EweAcT (arXiv:2608.09943), including Figure 1
  decoded from the PDF for the per-class percentages
- Recherche Data Gouv Dataverse API records for EweAcT (`doi:10.57745/AM2AJI`) and
  CabriTrack (`doi:10.57745/CYF9QW`): licence, file list, sizes, full descriptions
- First 1,501 bytes of EweAcT's AX3 sample file, by HTTP range request, to confirm units
  and sample spacing
- PMC rendering of the CabriTrack Data in Brief article (one successful fetch; a second
  attempt hit a captcha)
- Crossref record for `10.1016/j.dib.2025.111431`
- UCI dataset pages and JSON API records for datasets 240, 231, 341 and 507, including the
  full `variable_info` text
- The published UCI HAR zip: directory listing, `README.txt` units paragraph, and
  `y_train.txt` / `y_test.txt` counted directly
- `cis.fordham.edu/wisdm/dataset.php` (full text) and the Kwapisz et al. 2010 PDF from the
  same site
- DANS Data Station Dataverse API record for `doi:10.17026/dans-zp6-fmna`; DataCite records
  for `10.17026/dans-xhn-bsfb` and `10.4121/uuid:2e08745c-...`
- `utwente.nl/en/eemcs/ps/research/dataset/`
- ORA record and GitHub README for CAPTURE-24
- GitHub README for MotionSense
- `uni-mannheim.de` RealWorld project and dataset pages
- Frontiers full text for `10.3389/fvets.2025.1630083`
- HTTP HEAD/range checks confirming the UCI 240, UCI 231, WISDM v1 and UCI 507 download
  URLs resolve

Seen only in search results, not opened:

- `sciencedirect.com` article pages (HTTP 403 to automated fetches throughout): Turner
  et al. 2022/2023, the lambing-detection paper and its "Data will be made available on
  request" statement, and the goat behaviour data paper `S2772694025000044`
- ACM Digital Library pages for Kamminga et al. (HTTP 403)
- The CSIRO Arm18/Arm20 preprints
- The *Tjotta accelerometer monitored lambing dataset* (Scientific Data), noted here only
  because it surfaced while checking Turner's data availability; it covers lambing events,
  not grazing behaviour

Figures marked "not stated" above were absent from every primary source I could open; they
are not omissions of searching.
