# Air Quality Forecast

## Overview
This project predicts PM2.5 concentrations in the SF Bay Area using an LSTM model trained on historical air quality and weather data. The model is trained on AWS SageMaker and deployed on Google Cloud Run, with data processing pipelines on AWS Fargate and automated scheduling with AWS EventBridge.

## Problem Statement
Existing air quality resources tend to focus on either broad regional forecasts or real-time local measurements, leaving a crucial gap: the ability to forecast air quality at a local level. 

## Solution
This project addresses the problem by building a comprehensive air quality prediction application specifically for the Bay Area.  Instead of relying on generalized regional data, this application leverages the power of Long Short-Term Memory (LSTM) networks to predict PM2.5 concentrations at a local level.

## Project Structure
The project is organized into four key parts:

- **Root Directory**: Contains the code for the user-facing prediction service, including the main application logic (`app.py`) and the user interface (`templates/index.html`).
- **prepare/**: Contains the data pipeline, deployed as a serverless task on AWS Fargate.
- **train/**: Encompasses the model training workflow, executed on AWS SageMaker.
- **lambda/**: Acts as a trigger for the model retraining process, deployed as an AWS Lambda function.

## Deployment
The entire application is containerized using Docker and deployed using a CI/CD pipeline with GitHub Actions.

## Future Development
Plans include:

- Adding real-time traffic data as a feature
- Resolving air quality data inconsistency

## Links
- [Medium Blog Post](https://medium.com/@sang.ahn.94/forecasting-air-quality-pm-2-5-in-the-sf-bay-area-with-lstm-aws-and-google-cloud-34be04215a05)
- [Demo Video](https://youtu.be/zW5Wst9unVA)