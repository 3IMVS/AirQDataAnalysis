import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np
import csv
import datetime as dt
import matplotlib.dates as mdates
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import metrics
from math import sqrt
import statsmodels.api as sm

plt.style.use('seaborn-pastel')


def abline(slope, intercept, step):
    """Plot a line from slope and intercept
    does the same as predict"""
    axes = plt.gca()
    x_vals = np.arange(0, step, 1)
    y_vals = intercept + slope * x_vals
    # plt.plot(x_vals, y_vals, label="slope")


def eCO2LinearRegression(parsedCSV, dataCapPercent=100):
    """linear regression of eCO2 data eCO2=f(time)"""

    # Parse orddict to lists
    time = []
    eCO2 = []

    # converts precents to number of training data samples needed
    dataCounter = round(dataCapPercent*len(parsedCSV)/100)
    for row in parsedCSV:
        date_time_obj = dt.datetime.strptime(row["time"], '%Y-%m-%d %H:%M:%S')

        timeobj = int(date_time_obj.strftime('%Y%m%d%H%M%S'))
        time.append((int(timeobj)))
        eCO2.append(float(row["eco2"]))

    # select large chunk of data to train model.
    x = np.array(time[0:dataCounter]).reshape((-1, 1))
    y = np.array(eCO2[0:dataCounter])

    # create model from time and eCO2
    model = LinearRegression().fit(x, y)

    # select untrained data
    x_new = np.array(time[dataCounter:]).reshape((-1, 1))

    predictionData = model.predict(x_new)

    # evaluate model and prediction.
    r_sq = model.score(x, y)
    rmse = mean_squared_error(
        eCO2[dataCounter:], predictionData, squared=False)

    # merge predicteded curve with data for graphs
    predictionData = mixPredictionWithRealData(
        dataCapPercent, predictionData, eCO2)

    LinearRegressionData = {
        "predictor": predictionData,
        "r_squared": r_sq,
        "rmse": rmse

    }
    return LinearRegressionData


def eCO2MultipleRegression(parsedCSV, dataCapPercent=100):
    """linear regression of eCO2 data eCO2=f(time)"""

    eCO2 = []
    timeTempHumidityPress = []
    # converts precents to number of training data samples needed
    dataCounter = round(dataCapPercent*len(parsedCSV)/100)

    for row in parsedCSV:
        date_time_obj = dt.datetime.strptime(row["time"], '%Y-%m-%d %H:%M:%S')
        timeobj = int(date_time_obj.strftime('%Y%m%d%H%M%S'))
        eCO2.append(float(row["eco2"]))
        timeTempHumidityPress.append([(int(timeobj)), float(row["temp"]), float(
            row["humidity"]), float(row["press"])])  # , int(row["tvoc"]) #per daug korealiuoja su eCO2 todel apanikintas

    x = np.array(timeTempHumidityPress)
    y = np.array(eCO2)

    # Takes date in range of [0..dataCap%]
    model = LinearRegression().fit(x[0:dataCounter], y[0:dataCounter])

    r_sq = model.score(x[0:dataCounter], y[0:dataCounter])
    # predicts from [%dataCap..Final_value]

    predictionData = model.predict(x[dataCounter:])

    # formats prediction curve to fit x dimention, based on data percentage
    rmse = mean_squared_error(
        eCO2[dataCounter:], predictionData, squared=False)

    predictionData = mixPredictionWithRealData(
        dataCapPercent, predictionData, eCO2)

    MultipleRegression = {
        "predictor": predictionData,
        "r_squared": r_sq,
        "rmse": rmse

    }
    return MultipleRegression


def mixPredictionWithRealData(realDataPercent, predictionData, realData):
    """Mixes prediction with real data for graph to show where real data ends"""
    dataCounter = round(realDataPercent*len(realData)/100)
    for i in (range(dataCounter)):
        predictionData = np.insert(predictionData, i, realData[i])
    return predictionData


def eCO2PolynomialRegression(parsedCSV, dataCapPercent=100):

      # Parse orddict to lists
    eCO2 = []
    timeTempHumidityPress = []
    dataCounter = round(dataCapPercent*len(parsedCSV)/100)

    for row in parsedCSV:
        date_time_obj = dt.datetime.strptime(row["time"], '%Y-%m-%d %H:%M:%S')
        timeobj = int(date_time_obj.strftime('%Y%m%d%H%M%S'))
        eCO2.append(float(row["eco2"]))
        timeTempHumidityPress.append([(int(timeobj)), float(row["temp"]), float(
            row["humidity"]), float(row["press"])])

    # Provide data
    x = np.array(timeTempHumidityPress)
    y = np.array(eCO2)

    # Transform input data
    x_ = PolynomialFeatures(degree=2, include_bias=False,
                            interaction_only=True).fit_transform(x[0:dataCounter])

    # Create a model and fit it
    model = LinearRegression().fit(x_, y[0:dataCounter])

    # Transform input data
    x_new_ = PolynomialFeatures(degree=2, include_bias=False,
                                interaction_only=True).fit_transform(x[dataCounter:])

    # Predict

    predictionData = model.predict(x_new_)

    # evaluate

    r_sq = model.score(x_, y[0:dataCounter])

    rmse = mean_squared_error(
        eCO2[dataCounter:], predictionData, squared=False)

    predictionData = mixPredictionWithRealData(
        dataCapPercent, predictionData, eCO2)

    PolynomialRegression = {
        "predictor": predictionData,
        "r_squared": r_sq,
        "rmse": rmse

    }
    return PolynomialRegression


def parseAirQ_CSV(CSV_File, dataCount=1000000):
    """ Parses air quality monitor csv file to dictionary? which allows to handle
    data globaly
    """
    csvDictList = []
    with open(CSV_File) as csv_file:
        fieldNames = ["time", "temp", "humidity", "press", "eco2", "tvoc"]
        csv_reader = csv.DictReader(csv_file, fieldnames=fieldNames)
        line_count = 0
        for row in csv_reader:
            line_count += 1
            csvDictList.append(row)
            if line_count == dataCount:
                break

    print(f'Processed {line_count} lines.')
    return csvDictList


def plotAirQ_Graphs(parsedCSV, prediction, dataCapPercent):
    """ Plot temperature, humidity, pressure, eCO2(with prediction line) graphs
    """
    # Parse orddict to lists
    time = []
    temp = []
    humidity = []
    pressure = []
    eCO2 = []
    tvoc = []
    dataCounter = round(dataCapPercent*len(parsedCSV)/100)
    rsqr = prediction["r_squared"]
    rmse = prediction["rmse"]
    for row in parsedCSV:
        date_time_obj = dt.datetime.strptime(row["time"], '%Y-%m-%d %H:%M:%S')
        time.append(date_time_obj)
        temp.append(float(row["temp"]))
        humidity.append(float(row["humidity"]))
        eCO2.append(float(row["eco2"]))
        pressure.append(float(row["press"]))
        tvoc.append(int(row["tvoc"]))

    fig, ax = plt.subplots(5, figsize=(16, 9), sharex=True)

    ax[1].set_yticks(np.arange(round(min(humidity)),
                               round(max(humidity))+1, 5))
    ax[0].set_yticks(np.arange(round(min(temp)), round(max(temp))+1, 1))

    ax[3].set_yticks(np.arange(round(min(pressure)),
                               round(max(pressure))+1, 5))

    ax[4].set_yticks(np.arange(min(eCO2), max(eCO2)+1, 500))

    ax[0].plot(time, temp, label="temperature", color='orangered')
    ax[1].plot(time, humidity, label="humidity", color="lightskyblue")
    ax[2].plot(time, tvoc, label="tvoc", color="mediumpurple")
    ax[3].plot(time, pressure, label="pressure", color="m")
    ax[4].plot(time, eCO2, label="eCO2", color="gold")
    ax[4].plot(time, prediction["predictor"], label=f"\nR2:{rsqr:0.4f}\nRMSE:{rmse:0.0f}",
               color="dimgrey", linestyle=':')

    ax[4].annotate('prediction start',
                   xy=(time[dataCounter], eCO2[dataCounter]
                       ), xycoords='data',
                   xytext=(-20, 20), textcoords='offset pixels',
                   horizontalalignment='right',
                   verticalalignment='bottom',
                   arrowprops=dict(arrowstyle="->"))

    ax[0].title.set_text('Temperature')
    ax[1].title.set_text('Humidity')
    ax[2].title.set_text('tvoc')
    ax[3].title.set_text('Pressure')
    ax[4].title.set_text('eCO2 prediction')

    plt.tight_layout()
    # extra = Rectangle((0, 0), 1, 1, fc="w", fill=False,
    #                   edgecolor='none', linewidth=0)
    ax[4].legend(loc="lower left")
    plt.gcf().autofmt_xdate()
    # plt.legend()
    plt.show()


trainingDataPercentage = 65
parsedData = parseAirQ_CSV("dataLongFixed.csv")

predictLinear = eCO2LinearRegression(parsedData, trainingDataPercentage)

predictMultiLinear = eCO2MultipleRegression(parsedData, trainingDataPercentage)

predictPolynomial = eCO2PolynomialRegression(
    parsedData, trainingDataPercentage)

plotAirQ_Graphs(parsedData, predictLinear, trainingDataPercentage)
plotAirQ_Graphs(parsedData, predictMultiLinear, trainingDataPercentage)
plotAirQ_Graphs(parsedData, predictPolynomial, trainingDataPercentage)
