# Public data licences

The text models are trained and evaluated on public corpora with human readability
judgements. The files are downloaded by `scripts/download_public_data.py` into the
git-ignored `data/public/` folder and are never committed, copied into images or
redistributed. They are general English, not technical lecture material.

## CLEAR corpus (CommonLit Ease of Readability)

- File: `CLEAR_corpus_final.xlsx`
- Source URL: https://raw.githubusercontent.com/scrosseye/CLEAR-Corpus/main/CLEAR_corpus_final.xlsx
- Licence: CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)
- Accessed: 2026-10-23
- SHA-256: `e87a19a29c12a441e0c4f11a4e49897ff24ac7dff240ee7397283a050a902cc7`
- Rows used: 4724
- Target: BT_easiness (Bradley-Terry ease, teacher judgements); higher = easier
- Usage restrictions: Research and non-commercial use only; attribution and share-alike apply. The data is never committed, never copied into images and never redistributed; weights trained on it carry the non-commercial restriction (state it in the model card).

## OneStopEnglish corpus

- File: `onestop.zip`
- Source URL: https://codeload.github.com/nishkalavallabhi/OneStopEnglishCorpus/zip/refs/heads/master
- Licence: CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0/)
- Accessed: 2026-10-23
- SHA-256: `fea6300ed18e22b2009bb79aac949cb5cf6404d75ae8b1734daa28d8fcd9e061`
- Rows used: 567
- Target: reading level Ele=0, Int=1, Adv=2 (three versions of each article); higher = harder
- Usage restrictions: Attribution and share-alike apply. The data is never committed and never copied into images; derived artefacts must keep the attribution.
