# Music Recommendation Statistics Project

A Python analysis project that models music-rating behavior and evaluates simple recommendation strategies with probability, Bayesian reasoning, user variability modeling, and offline Monte Carlo replay.

This project was built for **CMPE343 - Introduction to Statistics for Computer Engineering Students**. It uses anonymized/expanded music rating data and track metadata to explore how statistical methods can support recommender-system decisions.

The repository also includes the original course report (`report.pdf`) and a small group-rating sample (`data/group_ratings.csv`) for context.

## Features

- Conditional probability analysis for questions such as `P(5-star rating | artist)`, `P(5-star rating | release year)`, and `P(5-star rating | explicit flag)`.
- Laplace smoothing for sparse feature values.
- Bayesian inversion to estimate which feature values are most associated with 5-star ratings.
- Feature-interaction analysis, such as mood and timbre combinations.
- User variability modeling with time-to-favorite calculations.
- Geometric and beta-geometric parameter estimation.
- Welch's t-test, Mann-Whitney U test, and D'Agostino-style normality checks implemented directly in Python.
- Popularity-based and personalized recommendation strategies.
- Offline Monte Carlo replay comparing popular and personalized recommenders with metrics such as Hit@K, average rating, and time-to-5.

## Tech Stack

- Python 3.10+
- pandas
- NumPy
- CSV datasets for ratings and track metadata

## Usage Flow

The project is split into four executable scripts:

- `src/part1.py`: conditional probability and Bayesian analysis
- `src/part2.py`: user variability and hypothesis testing
- `src/part3.py`: recommender design demo
- `src/part4.py`: Monte Carlo recommender evaluation

Run scripts from the project root so the relative `data/` paths resolve correctly.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run

```bash
python src/part1.py
python src/part2.py
python src/part3.py
```

Run a quick evaluation:

```bash
python src/part4.py --n-trials 20
```

Run the default full evaluation:

```bash
python src/part4.py
```

Optional Part 4 arguments:

```bash
python src/part4.py --topk 3 --n-trials 100 --ratings-path data/enlarged_user_ratings.csv
```

## Project Structure

```text
.
├── data/
│   ├── enlarged_user_ratings.csv
│   ├── group_ratings.csv
│   ├── tracks.csv
│   └── user_ratings.csv
├── report.pdf
├── src/
│   ├── part1.py
│   ├── part2.py
│   ├── part3.py
│   └── part4.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Skills Demonstrated

- Probability modeling and Bayesian interpretation
- Statistical hypothesis testing
- Maximum likelihood estimation
- Offline recommender-system evaluation
- Data wrangling with pandas
- NumPy-based numerical computation
- Modular Python scripting
- Clear experiment structure and reproducible run commands


