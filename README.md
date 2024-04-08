# Shatranj---Chess-Tutor

![chess](https://github.com/bhavyamistry/Shatranj-Chess-Tutor/assets/58860047/1506ded6-bb10-4f9d-a215-299cf7af162e)


<img src="./images/cycle.gif" alt="vector" align="center">

## Project Overview

This project focuses on analyzing bicycle sales data to understand customer preferences based on various factors such as geographical location, occupation, gender, age, commute distance, and vehicle ownership. By visualizing this data, sellers can gain insights into customer demographics and tailor their offerings to better meet customer needs.

## Problem Statement

The bicycle industry faces challenges in understanding trends in bicycle sales based on location and customer demographics. Lack of insight into customer preferences hinders sellers' ability to effectively target their offerings. This project aims to address this issue by analyzing bicycle sales data from three regions: North America, the Pacific, and Europe.

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

