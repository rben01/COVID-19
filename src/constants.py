from pathlib import Path


class Paths:
    ROOT = Path("..")


class Columns:
    LATITUDE = "Lat"
    LONGITUDE = "Long"
    STATE = "Province/State"
    COUNTRY = "Country/Region"
    LOCATION_NAME = "Location"
    DATE = "Date"
    CASE_COUNT = "Cases"
    CASE_TYPE = "Case Type"


class CaseTypes:
    CONFIRMED = "Confirmed"
    RECOVERED = "Recovered"
    DEATHS = "Deaths"
    MORTALITY = "Mortality"


class Locations:
    WORLD = "World"
    WORLD_MINUS_CHINA = "Non-China"
    CHINA = "China"
    US = "US"
    UK = "United Kingdom"
