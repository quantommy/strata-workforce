# Strata

A workforce intelligence prototype for Hackathon Track A.

Company overview shows risk distributions, department priorities and role/location analysis.
Career workspace combines a submitted profile, adjustable role rankings and learning workflows.

## Model and data

The selected enriched MLP trains from scratch. Validation accuracy is 36.1% and macro F1 is 0.350.
Scores are exploratory, uncalibrated class scores. The 500 included rows are demo job profiles.
Demo departments derive from titles. Optional employee uploads supply actual IDs and departments.
Career rankings use salary, authored skill checklists, outlook and remote preferences.
The small demo data is not current salary market research or validated career outcomes.

## Streamlit Community Cloud

Deploy app.py from the main branch and choose Python 3.12.
Runtime dependencies match the model export. Inference needs no training or external downloads.
The repository contains no credentials, uploaded employee records, notebook, or challenge PDF.

## Local preview

Install requirements.txt in Python 3.12 and run python -m streamlit run app.py.

## Attribution

Modified O*NET 24.2 (2020), USDOL/ETA, CC BY 4.0 supplies occupational descriptors.
Historical US definitions are not validated worldwide forecasts. Local title mappings are modifications.
