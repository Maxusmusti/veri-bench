# veri-bench
Evaluating the quality of verifiers (for math currently)

## Description
(add description here)

## Evaluation and Scoring
(explain scoring here)

## Setup and Installation
```
pip install -r requirements.txt
```

## Usage
Add a directory for your verifier under `verifiers`. This directory must include a `verifier.py` file, which should in turn contain a function with the following header:
```
def verify(generated, gt) -> bool
```
