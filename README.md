# Project

This is a standalone package created to simulate changes in the size of a given
population.

## Project Structure

See PyPa
sample [project](https://github.com/pypa/sampleproject/commit/d4ee05fdc03e848ed6e7065d8fe8e833a3c8c0b2)
.

## Code Style
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Data
All data provided is publicly available, see:

### Population Pyramid
 - UK 2011 [Census](https://www.ons.gov.uk/peoplepopulationandcommunity/populationandmigration/populationestimates/datasets/2011censuspopulationestimatesbysingleyearofageandsexforlocalauthoritiesintheunitedkingdom)

### Mortality Rates
 UK Life Expectancies (12 Jan 2022) (cohorts):
 - [Wales](https://www.ons.gov.uk/peoplepopulationandcommunity/birthsdeathsandmarriages/lifeexpectancies/datasets/mortalityratesqxprincipalprojectionwales)
 - [Scotland](https://www.ons.gov.uk/peoplepopulationandcommunity/birthsdeathsandmarriages/lifeexpectancies/datasets/mortalityratesqxprincipalprojectionscotland)
 - [Northern Ireland](https://www.ons.gov.uk/peoplepopulationandcommunity/birthsdeathsandmarriages/lifeexpectancies/datasets/mortalityratesqxprincipalprojectionnorthernireland)
 - [England](https://www.ons.gov.uk/peoplepopulationandcommunity/birthsdeathsandmarriages/lifeexpectancies/datasets/mortalityratesqxprincipalprojectionengland)

The data is truncated so that everyone who has their age going beyond 100 has 100% mortality rate.

### Birth Rates
 - England & Wales live births by [Nomis](https://www.nomisweb.co.uk/query/select/getdatasetbytheme.asp?theme=73) & [ONS](https://www.ons.gov.uk/peoplepopulationandcommunity/birthsdeathsandmarriages/livebirths/adhocs/12417numberoflivebirthsbysexenglandandwales1982to2019) (separately) 2021
 - Scotland Birth Time Series [Data](https://www.nrscotland.gov.uk/statistics-and-data/statistics/statistics-by-theme/vital-events/births/births-time-series-data) 2021
 - Northen Ireland Birth [Statistics](https://www.nisra.gov.uk/publications/birth-statistics) 2021

Note that ONS provides data for residents inside England and Wales only as opposed to Nomis.
However, the difference is negligible.

Also, all data is combined into a single table at the moment.

#### Note
Mortality rates tables require a password to get access to, this can be fixed by
converting the document to `*.xlsx`.
