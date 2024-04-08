# Shatranj-Chess Openings Analyzer

![chess](https://github.com/bhavyamistry/Shatranj-Chess-Tutor/assets/58860047/1506ded6-bb10-4f9d-a215-299cf7af162e)


## Project Overview

The Chess Openings Analyzer project leverages machine learning (ML) techniques to analyze the success rates of various chess openings. By incorporating the latest advances in ML and data mining, such as deep learning algorithms and neural networks, this project offers several advantages over current state-of-the-art solutions.

Using machine learning algorithms like logistic regression, the project predicts game results based on openings, ELO ratings, and other features in the dataset. Additionally, clustering techniques are employed to analyze the results of different openings, providing players with valuable insights into strategic decision-making.

## Problem Statement

The objective of this data mining project is to predict the success of various chess openings using historical game data. The goal is to leverage machine learning techniques to analyze patterns within the dataset and gain insights into the effectiveness of different opening strategies.

## Dataset

The dataset used in this project is sourced from Kaggle and includes information on bicycle purchases, including geographical location, occupation, gender, age, commute distance, and vehicle ownership.

[Dataset Link](https://www.kaggle.com/datasets/heeraldedhia/bike-buyers)

## Libraries Used

The following libraries were used in this project: <br>
![Numpy](https://img.shields.io/badge/Numpy-777BB4?style=for-the-badge&logo=numpy&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-2C2D72?style=for-the-badge&logo=pandas&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-20BEFF?style=for-the-badge&logo=Kaggle&logoColor=white)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-FF5733?style=for-the-badge&logo=matplotlib&logoColor=white)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-008000?style=for-the-badge&logo=seaborn&logoColor=white)](https://seaborn.pydata.org/)

## Project Demonstration

Click the link below to watch a demonstration of the project:

[Project Video](https://www.youtube.com/watch?v=YboXwoAtaik)

## Conclusion

By analyzing bicycle sales data, we can gain valuable insights into customer preferences and trends. This information can help sellers make informed decisions and tailor their offerings to better meet customer needs, ultimately improving sales and customer satisfaction.

## Instructions

1. CREATE AN .env file
```
python3 -m venv <virtual-environment-name>
```
2. ADD THE FOLLOWING LINES IN IT

```
FLASK_APP=flask_app.py
FLASK_DEBUG=1
FLASK_RUN_PORT = 3000
APP_SECRET_KEY = IOAJODJAD89ADYU9A78YGD
```
3. THEN RUN THE FOLLOWING IN THE TERMINAL WITHIN THE ENVIRONMENT
```
pip3 install -r requirements.txt
```
4. TO RUN THE SERVER USE
```
flask run
```
OR
```
python3 flask_app.py
```

