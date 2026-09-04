# Data card

## What the dataset is

`data/exam_results_cleaned_final.csv` holds 122 rows. Each row is one student's
written answer to one question, together with the mark a single instructor
awarded it.

| column | type | description |
|---|---|---|
| `question_number` | integer, 1 to 4 | which of the four exam questions the answer responds to |
| `transcribed_text` | string | the text of the handwritten answer |
| `normalized_mark` | float, 1.0 to 4.0 in half steps | the mark awarded, on a shared 4-point scale |

## Provenance

The answers come from an internal examination in BCE404, an introductory
biosensor course in the B.Tech Biotechnology programme at Amity University
Kolkata. Papers were written by hand in answer booklets and marked by hand by
the course instructor.

The four questions are identifiable from the answer text.

| column value | topic |
|---|---|
| 1 | sensitivity and specificity of a TMB colorimetric sensor against HRP |
| 2 | biofunctionalization of gold nanoparticles |
| 3 | physicochemical properties of nanomaterials in biosensing |
| 4 | limit of detection and its derivation from a calibration line |

The exact sitting, the marks each question carried on the paper, and the raw
booklet images are not recorded in this repository. That is a provenance gap.
Anyone extending this work should write the exam date, the paper, and a pointer
to the source images into the repository before adding rows.

The booklets were photographed page by page and transcribed with a
vision-language model, then the marks written on each answer were read from the
same images. Transcription and mark extraction were both automated. Neither was
verified against the original booklets answer by answer, so an unknown fraction
of the rows carry transcription noise, and an unknown fraction carry a
misread mark. This is a real limitation on every number in this repository and
it is not quantified anywhere.

Rows with blank answers, unreadable text, or a mark outside 1.0 to 4.0 were
dropped during cleaning. Exact duplicate answer texts were dropped. The
122 rows that remain contain no duplicate texts.

`normalized_mark` takes the seven values 1.0, 1.5, 2.0, 2.5, 3.0, 3.5 and 4.0.
The column name says a normalization was applied. What the raw marks were and
what transformation produced this column are not recorded anywhere in the
repository, so the mapping cannot be reconstructed from what is here.

Treat the column as an ordinal grade on a shared scale and nothing more. In
particular, a 3.0 on one question is not established as the same standard as a
3.0 on another. This matters, because the strongest signal in the dataset is
question identity.

## Anonymization

The CSV carries three columns and none of them is an identifier. Student names,
enrollment numbers, serial numbers, and the instructor's name were never
written into it. The upstream transcription step did extract names and
enrollment numbers from the booklet cover pages. Those intermediate files stay
outside this repository and are not published.

Answer text can still carry incidental identifying content, for example a
student writing their own name inside an answer. A pattern scan across all 122
texts for enrollment-number formats and for the strings `name:`, `roll no` and
`enroll` returns no matches. That scan catches formatted identifiers. A name
written as a bare word would pass it, and the texts have not been audited
line by line.

## Consent and reuse

The answers were produced as coursework in a formal assessment. Students did
not give consent for their work to be published or redistributed. The rows are
short factual answers to technical questions, they carry no personal
information, and they are stripped of identifiers.

Anyone considering reusing this data outside the terms of the licence should
weigh that consent gap first. Do not treat this as a general-purpose
educational corpus.

## Licence

Code in this repository is MIT (`LICENSE`). The data is CC-BY-4.0
(`DATA_LICENSE`). The licence covers redistribution. It does not settle the
consent question above, which is a separate matter.

## Synthetic file

`data/synthetic_exam_data_categorical.csv` holds 400 machine-generated answers
written to bulk up the early fine-tuning attempts. No model reported in
`results/` touches it. It is kept because `failed_attempts/` refers to it and
those scripts should stay runnable. Do not use it to train or evaluate
anything.

## Known limitations

- 122 rows across 4 questions, roughly 30 answers per question. Every subgroup
  analysis in this repository runs on 20 to 35 answers.
- One instructor, one exam, one cohort, one course. Nothing here transfers to
  another grader or another paper without new data.
- Question 4 holds 26 Low marks out of 28. That question carries almost no
  variation to model.
- Marks and transcriptions both come from automated extraction with no
  measured error rate.
- The transformation behind `normalized_mark` is undocumented.
- The 4 questions differ in difficulty, in marking standard, and in expected
  answer length. Any pooled analysis across questions inherits all three.
