# Original UI testing documents

These fictional financial narratives were written for software acceptance testing.
They are not genuine earnings releases, gold sentiment labels or model-accuracy
benchmarks. Filenames describe the intended content; tests must use actual model
predictions without requiring a particular sentiment distribution.

- `positive_earnings.txt`: ten original financial sentences.
- `negative_earnings.txt`: ten original financial sentences.
- `mixed_earnings.txt`: twelve original sentences with figures, percentages,
  abbreviations, neutral statements and negation.
- `sample_earnings.pdf`: one page of original text-based financial commentary,
  generated locally with the already installed PyMuPDF library.
- `image_only_earnings.pdf`: one page containing only a generated solid image,
  for testing the unsupported-OCR error. It contains no extractable text.

All files are small and original. Authentic Microsoft/Apple documents remain in
ignored `data/raw/` and are not copied into these fixtures. Generated predictions
and downloaded reports remain under ignored `outputs/phase6/`.
