# Channel Selection for Real CHB-MIT EEG

Local EDF files only were used. No synthetic EEG and no downloaded files were included.

## Subjects
- chb01: 20 EDF files
- chb02: 9 EDF files
- chb03: 13 EDF files
- chb05: 14 EDF files
- chb08: 14 EDF files

## Harmonization Rule
- Channel labels were normalized by trimming whitespace, uppercasing, and removing MNE duplicate suffixes such as `-0` and `-1`.
- When an EDF contained duplicate channel labels, the first occurrence was retained and later duplicate occurrences were excluded.
- The final feature matrix uses the common channel set present in every selected EDF file.

## Final Common Channel Set (22 channels)
1. `FP1-F7`
2. `F7-T7`
3. `T7-P7`
4. `P7-O1`
5. `FP1-F3`
6. `F3-C3`
7. `C3-P3`
8. `P3-O1`
9. `FP2-F4`
10. `F4-C4`
11. `C4-P4`
12. `P4-O2`
13. `FP2-F8`
14. `F8-T8`
15. `T8-P8`
16. `P8-O2`
17. `FZ-CZ`
18. `CZ-PZ`
19. `P7-T7`
20. `T7-FT9`
21. `FT9-FT10`
22. `FT10-T8`

## Duplicate Channels Removed
- `T8-P8`: duplicate occurrence removed in 70 EDF files

## Channels Excluded From Final Common Set
- None beyond duplicate-channel removal; all unique channels were common.

## Per-Recording Notes
- chb01 `chb01_01.edf`: duplicates removed = T8-P8-1->T8-P8; excluded = none
- chb01 `chb01_02.edf`: duplicates removed = T8-P8-1->T8-P8; excluded = none
- chb01 `chb01_03.edf`: duplicates removed = T8-P8-1->T8-P8; excluded = none
- chb01 `chb01_04.edf`: duplicates removed = T8-P8-1->T8-P8; excluded = none
- chb01 `chb01_05.edf`: duplicates removed = T8-P8-1->T8-P8; excluded = none
- chb01 `chb01_06.edf`: duplicates removed = T8-P8-1->T8-P8; excluded = none
- chb01 `chb01_14.edf`: duplicates removed = T8-P8-1->T8-P8; excluded = none
- chb01 `chb01_15.edf`: duplicates removed = T8-P8-1->T8-P8; excluded = none
- chb01 `chb01_16.edf`: duplicates removed = T8-P8-1->T8-P8; excluded = none
- chb01 `chb01_17.edf`: duplicates removed = T8-P8-1->T8-P8; excluded = none
- chb01 `chb01_18.edf`: duplicates removed = T8-P8-1->T8-P8; excluded = none
- chb01 `chb01_19.edf`: duplicates removed = T8-P8-1->T8-P8; excluded = none
- chb01 `chb01_20.edf`: duplicates removed = T8-P8-1->T8-P8; excluded = none
- chb01 `chb01_21.edf`: duplicates removed = T8-P8-1->T8-P8; excluded = none
- chb01 `chb01_22.edf`: duplicates removed = T8-P8-1->T8-P8; excluded = none
- chb01 `chb01_23.edf`: duplicates removed = T8-P8-1->T8-P8; excluded = none
- chb01 `chb01_24.edf`: duplicates removed = T8-P8-1->T8-P8; excluded = none
- chb01 `chb01_25.edf`: duplicates removed = T8-P8-1->T8-P8; excluded = none
- chb01 `chb01_26.edf`: duplicates removed = T8-P8-1->T8-P8; excluded = none
- chb01 `chb01_27.edf`: duplicates removed = T8-P8-1->T8-P8; excluded = none
- chb02 `chb02_01.edf`: duplicates removed = T8-P8-1->T8-P8; excluded = none
- chb02 `chb02_15.edf`: duplicates removed = T8-P8-1->T8-P8; excluded = none
- chb02 `chb02_16.edf`: duplicates removed = T8-P8-1->T8-P8; excluded = none
- chb02 `chb02_16+.edf`: duplicates removed = T8-P8-1->T8-P8; excluded = none
- chb02 `chb02_17.edf`: duplicates removed = T8-P8-1->T8-P8; excluded = none
- chb02 `chb02_18.edf`: duplicates removed = T8-P8-1->T8-P8; excluded = none
- chb02 `chb02_19.edf`: duplicates removed = T8-P8-1->T8-P8; excluded = none
- chb02 `chb02_20.edf`: duplicates removed = T8-P8-1->T8-P8; excluded = none
- chb02 `chb02_21.edf`: duplicates removed = T8-P8-1->T8-P8; excluded = none
- chb03 `chb03_01.edf`: duplicates removed = T8-P8-1->T8-P8; excluded = none
- chb03 `chb03_02.edf`: duplicates removed = T8-P8-1->T8-P8; excluded = none
- chb03 `chb03_03.edf`: duplicates removed = T8-P8-1->T8-P8; excluded = none
- chb03 `chb03_04.edf`: duplicates removed = T8-P8-1->T8-P8; excluded = none
- chb03 `chb03_05.edf`: duplicates removed = T8-P8-1->T8-P8; excluded = none
- chb03 `chb03_31.edf`: duplicates removed = T8-P8-1->T8-P8; excluded = none
- chb03 `chb03_32.edf`: duplicates removed = T8-P8-1->T8-P8; excluded = none
- chb03 `chb03_33.edf`: duplicates removed = T8-P8-1->T8-P8; excluded = none
- chb03 `chb03_34.edf`: duplicates removed = T8-P8-1->T8-P8; excluded = none
- chb03 `chb03_35.edf`: duplicates removed = T8-P8-1->T8-P8; excluded = none
- chb03 `chb03_36.edf`: duplicates removed = T8-P8-1->T8-P8; excluded = none
- chb03 `chb03_37.edf`: duplicates removed = T8-P8-1->T8-P8; excluded = none
- chb03 `chb03_38.edf`: duplicates removed = T8-P8-1->T8-P8; excluded = none
- chb05 `chb05_01.edf`: duplicates removed = T8-P8-1->T8-P8; excluded = none
- chb05 `chb05_05.edf`: duplicates removed = T8-P8-1->T8-P8; excluded = none
- chb05 `chb05_06.edf`: duplicates removed = T8-P8-1->T8-P8; excluded = none
- chb05 `chb05_07.edf`: duplicates removed = T8-P8-1->T8-P8; excluded = none
- chb05 `chb05_12.edf`: duplicates removed = T8-P8-1->T8-P8; excluded = none
- chb05 `chb05_13.edf`: duplicates removed = T8-P8-1->T8-P8; excluded = none
- chb05 `chb05_14.edf`: duplicates removed = T8-P8-1->T8-P8; excluded = none
- chb05 `chb05_15.edf`: duplicates removed = T8-P8-1->T8-P8; excluded = none
- chb05 `chb05_16.edf`: duplicates removed = T8-P8-1->T8-P8; excluded = none
- chb05 `chb05_17.edf`: duplicates removed = T8-P8-1->T8-P8; excluded = none
- chb05 `chb05_18.edf`: duplicates removed = T8-P8-1->T8-P8; excluded = none
- chb05 `chb05_21.edf`: duplicates removed = T8-P8-1->T8-P8; excluded = none
- chb05 `chb05_22.edf`: duplicates removed = T8-P8-1->T8-P8; excluded = none
- chb05 `chb05_23.edf`: duplicates removed = T8-P8-1->T8-P8; excluded = none
- chb08 `chb08_02.edf`: duplicates removed = T8-P8-1->T8-P8; excluded = none
- chb08 `chb08_03.edf`: duplicates removed = T8-P8-1->T8-P8; excluded = none
- chb08 `chb08_04.edf`: duplicates removed = T8-P8-1->T8-P8; excluded = none
- chb08 `chb08_05.edf`: duplicates removed = T8-P8-1->T8-P8; excluded = none
- chb08 `chb08_10.edf`: duplicates removed = T8-P8-1->T8-P8; excluded = none
- chb08 `chb08_11.edf`: duplicates removed = T8-P8-1->T8-P8; excluded = none
- chb08 `chb08_12.edf`: duplicates removed = T8-P8-1->T8-P8; excluded = none
- chb08 `chb08_13.edf`: duplicates removed = T8-P8-1->T8-P8; excluded = none
- chb08 `chb08_14.edf`: duplicates removed = T8-P8-1->T8-P8; excluded = none
- chb08 `chb08_20.edf`: duplicates removed = T8-P8-1->T8-P8; excluded = none
- chb08 `chb08_21.edf`: duplicates removed = T8-P8-1->T8-P8; excluded = none
- chb08 `chb08_22.edf`: duplicates removed = T8-P8-1->T8-P8; excluded = none
- chb08 `chb08_23.edf`: duplicates removed = T8-P8-1->T8-P8; excluded = none
- chb08 `chb08_24.edf`: duplicates removed = T8-P8-1->T8-P8; excluded = none
