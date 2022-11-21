# Project

This is a standalone package created to simulate changes in the size of a given
population.

## Project Structure

See PyPa
sample [project](https://github.com/pypa/sampleproject/commit/d4ee05fdc03e848ed6e7065d8fe8e833a3c8c0b2)
.

## Data
All data provided is publicly available, see:

### Population Syramid
 - UK 2011 [Census](https://www.ons.gov.uk/census/2011census/2011censusdata).
### Mortality Rates
 - UK [Life Tables](https://www.ons.gov.uk/peoplepopulationandcommunity/birthsdeathsandmarriages/lifeexpectancies/datasets/lifetablesprincipalprojectionunitedkingdom) 2020.

The data is truncated so that everyone who has their age going beyond 100 has 100% mortality rate.
### Birth Rates
 - England and Wales [Life Tables](https://www.ons.gov.uk/peoplepopulationandcommunity/birthsdeathsandmarriages/livebirths/datasets/birthsummarytables) 2021;
 - Scotland Birth Time Series [Data](https://www.nrscotland.gov.uk/statistics-and-data/statistics/statistics-by-theme/vital-events/births/births-time-series-data) 2021;
 - Northen Ireland Birth [Statistics](https://www.nisra.gov.uk/publications/birth-statistics) 2020;

All data is combined into a single table at the moment.
#### Note
Mortality rates tables require a password to get access to, this can be fixed by
converting the document to `*.xlsx`.