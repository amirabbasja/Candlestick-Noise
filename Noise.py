from random import randint, randrange
import string
from time import time
from tkinter.tix import Tree
from turtle import width
import pandas as pd
from plotly.subplots import make_subplots
import numpy as np
import matplotlib.pyplot as plt
from scipy import rand
from tqdm import tqdm
import mplfinance as mplf
import plotly.graph_objects as go
# In this script we add noise to candles. There are different ways of
# adding noise to candles and all the different methods are in the Noise
# class.
def getCandles(fileLocation : string) -> pd.DataFrame():
    # This fuunction reads a dataframe from a csv file
    return pd.read_csv(fileLocation)

def plotDataset(df:pd.DataFrame, method = "mplfinance") -> None:
    """
    The function to plot candlestick data.
    
    Arguments
    ---------
    df: pd.DataFrame: The dataframe containing open, high, close and low of candles in each column. 
        The dataframe has to contain a "date" column representing the open time of each cnadle. the
        dataframe can contain "close_time" and "volume" column but it is not necessary to provide 
        them.
    method: string: The method to plot the data. (Choose between "mplfinance" or "plotly")

    Returns
    -------
    None

    """
    
    if method == "mplfinance":
        import mplfinance as mplf
        
        # Setting the date column as index
        df.index = pd.DatetimeIndex(df['date'])

        mplf.plot(
            df,
            style="charles",
            type="candle",
            volume=False , # if "volume" not in df.columns else True, 
            title={"title": "Candlestick data"},
            tight_layout=True,
        )

    elif method == "plotly":
        import plotly.graph_objects as go   

        noisedChart = go.Candlestick(
                x=df["date"],
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name = "Candlestick data",
                # increasing={'line': {'color': 'blue'}},
                # decreasing={'line': {'color': 'purple'}},
                )

        layout = go.Layout(
            title = 'Overview',
            xaxis = go.XAxis(
                title = 'Date',
                rangeslider_visible = False,
                showticklabels=False),
            yaxis = go.YAxis(
                title = 'Price',
                fixedrange = False
            )
        )

        fig = go.Figure(data=[noisedChart])

        fig.show()

    else:
        print("Error, Method only mplfinance or plotly are allowed")

def saveCandles(df:pd.DataFrame, location:string, saveImage = False, method = "mplfinance") -> None:
    """
    The function to save the candlestick data to a pickle file.

    Arguments
    ---------
    df: pd.DataFrame: The dataframe containing open, high, close and low of candles in each column. 
        The dataframe has to contain a "date" column representing the open time of each cnadle. the
        dataframe can contain "close_time" and "volume" column but it is not necessary to provide 
        them.    
    location: string: The location of the file to save the data. It should contain file's name and 
    also it's extention.
    saveImage: boolean: If true the candlestick plot is saved as an image.
    method: string: The method to plot the data. (Choose between "mplfinance" or "plotly")

    Returns
    -------
    None

    Note
    ----
    * To use plotly as a method of saving images, you need to have kaleido package installed.

    """

    if saveImage:
        if method == "mplfinance":    
            import mplfinance as mplf
            
            # Setting the date column as index
            df.index = pd.DatetimeIndex(df['date'])

            mplf.plot(
                df,
                style="charles",
                type="candle",
                volume=False if "volume" not in df.columns else True, 
                title={"title": "Candlestick data"},
                tight_layout=True,
                savefig = location.replace(".pkl",".png")
            )
        elif method == "plotly":
            import plotly.graph_objects as go   

            noisedChart = go.Candlestick(
                    x=candles["date"],
                    open=candles['open'],
                    high=candles['high'],
                    low=candles['low'],
                    close=candles['close'],
                    name = "Candlestick data",
                    # increasing={'line': {'color': 'blue'}},
                    # decreasing={'line': {'color': 'purple'}},
                    )

            layout = go.Layout( margin=dict(l=20, r=20, t=20, b=20))
            fig = go.Figure(data=[noisedChart], layout=layout)
            fig.update_xaxes(rangeslider_visible=False,  title = location[location.rfind("/")+1:location.rfind(".")])
            fig.write_image(location.replace(".pkl",".png"), width=800, height=400)

    df.to_pickle(location)


class Noise():
    """
    The master class that contains the different methods of adding noise
    to dataset.
    """

    def __init__(self) -> None:
        """
        Initialize the class
        """
        pass

    def GaussianSeries(self, dataset: pd.Series, scale) -> pd.Series(dtype = "float64"):
        """
        Gets the gaussian noise for dataframe series. The mean value for calculation is
        the value of each row. The calculation formula of noise for a mean(u) and standard
        deviation (sigma) is stated below

        Noise(z) = exp(-(z-u)^2/2/sigma^2)/sigma/sqrt(2*pi)
        
        Arguments
        ---------
            dataset: a pandas series to add noise to
            scale: The standard deviation of the noise that will be added (In percentage units)

        
        Returns
        -------
            A pandas dataframe containing past data + Noise
        """
        scalePercent = scale / 100

        dataset = dataset.apply(lambda x: x + np.random.normal(loc = 0, scale = x * scalePercent, size = 1).item())

        return dataset

    def GaussianCnadles(self,
        time:pd.Series,
        open:pd.Series, 
        high:pd.Series, 
        close:pd.Series, 
        low:pd.Series, 
        volume:pd.Series, 
        close_time:pd.Series, 
        scale, 
        method: int
        ) -> pd.DataFrame(dtype="float64"):
        """
        Method that will add gaussian noise to a candlesstick dataset. There are two methods supported for now

        method1: The noise will be added to all the parameters of each candle. this will cause the candles to be
        non-continuous, i.e. The close of previous candle will not be equal to open of the current candle. (provided
        that the input candles are continous)

        method2: Adding the noise to high, low and the body of the candle. In this method open price of the first
        candle in dataset will be the same but the open of other candles will be cahnged with respect to the noise of 
        the candle's body (but the candles will stay continous).
        
        Arguments
        ---------
            open, high low, close: pd.Series: all pandas dataframes (or series)
            scale: [float]: The standard deviation of the noise that will be added respectively
            to body, upper wick and lower wick of the candles. if only one variable is passed 
            inside the array, it will be assigend to all the parts of the candle. (The scale has
            to be in percentage units)
        
        Returns
        -------
            A pandas dataframe with columns = ["time","open","high","low","close"]
        """
        _open, _high, _close, _low = None, None, None, None
        shape = time.shape[0]
        
        # Assigning the scales of noise in each part of candle
        if   len(scale) == 1:
            scale_body, scale_high, scale_low = 1 + scale[0] / 100, 1 + scale[0] / 100, 1 + scale[0] / 100
        elif len(scale) == 2:
            scale_body, scale_high, scale_low = 1 + scale[0] / 100, 1 + scale[1] / 100, 1 + scale[1] / 100
        elif len(scale) == 3:
            scale_body, scale_high, scale_low = 1 + scale[0] / 100, 1 + scale[1] / 100, 1 + scale[2] / 100

        # Convert series to dataframe
        frame = {"time":time,"open":open,"high":high,"low":low,"close":close}
        df = pd.DataFrame(frame)

        # We added this method to see if the prices contain negative value (which are not acceptable)
        # Only after the end of data generation we can know this, so we use a while loop 
        _continue = True

        while _continue:
            if method == 1:
                # The logic: In this method we do not care about not generating gaps. so we add noise
                # to the body of each candle but we keep the open price the same. this will avoid the 
                # noise to be accumulated and chart looks very much like the original (Without noise)
                
                df["Date"] = time
                df["open"] = open
                df["close"] = close
                df["high"] = high
                df["low"] = low
                df["volume"] = volume

                shape = time.shape[0]
                pert_body = np.random.normal(loc = 0, scale = 1, size = shape)
                pert_body /= max(pert_body)

                pert_upperWick = np.random.normal(loc = 0, scale = 1, size = shape)
                pert_upperWick /= max(pert_upperWick)

                pert_lowerWick = np.random.normal(loc = 0, scale = 1, size = shape)
                pert_lowerWick /= max(pert_lowerWick)

                temp = pd.DataFrame()
                temp["u1"] = df["high"] - df["close"]
                temp["u2"] = df["high"] - df["open"]
                temp["l1"] = df["close"] - df["low"]
                temp["l2"] = df["open"] - df["low"]

                _upperWickLen = temp[["u1","u2"]].max(axis=1)
                _lowerWickLen = temp[["u1","u2"]].max(axis=1)
                _bodyLen = close - open

                _body = _bodyLen * (1 + pert_body) * scale_body
                _high = _upperWickLen * (1 + pert_upperWick) * scale_high
                _low  = _lowerWickLen * (1 + pert_lowerWick) * scale_low
                _volume = volume * abs((_high - _low)/(high - low))

                df["_open"] = open
                df["_close"] = open + _body
                df["_high"] =  df[["_open","_close"]].max(axis=1) +_high
                df["_low"] =  df[["_open","_close"]].min(axis=1) +_low
                df["_volume"] = _volume

                frame = {
                    "date":time,
                    "open":df["_open"],
                    "high":df["_high"],
                    "low":df["_low"],
                    "close":df["_close"],
                    "volume":df["_volume"],
                    "close_time":close_time
                }

                returnDF = pd.DataFrame(frame)

                # Generate the data untill there are no negative values
                if not (returnDF.iloc[:,1:4] < 0).any().any():
                    _continue = False

            elif method == 2:
                # The logic: to keep the candles continouse and avoid making gamps, we use cumsum 
                # of the body + noise and assign the value to the candle's close. and then we assign
                # the close of the current candle to open of the next candle
                
                df["Date"] = time
                df["open"] = open
                df["close"] = close
                df["high"] = high
                df["low"] = low
                df["volume"] = volume

                shape = time.shape[0]
                pert_body = np.random.normal(loc = 0, scale = 1, size = shape)
                pert_body /= max(pert_body)

                pert_upperWick = np.random.normal(loc = 0, scale = 1, size = shape)
                pert_upperWick /= max(pert_upperWick)

                pert_lowerWick = np.random.normal(loc = 0, scale = 1, size = shape)
                pert_lowerWick /= max(pert_lowerWick)

                temp = pd.DataFrame()
                temp["u1"] = df["high"] - df["close"]
                temp["u2"] = df["high"] - df["open"]
                temp["l1"] = df["close"] - df["low"]
                temp["l2"] = df["open"] - df["low"]

                _upperWickLen = temp[["u1","u2"]].max(axis=1)
                _lowerWickLen = temp[["u1","u2"]].max(axis=1)
                _bodyLen = close - open

                _body = _bodyLen * (1 + pert_body) * scale_body
                _high = _upperWickLen * (1 + pert_upperWick) * scale_high
                _low  = _lowerWickLen * (1 + pert_lowerWick) * scale_low
                _volume = volume * abs((_high - _low)/(high - low))

                _close = np.cumsum(_body) + close.iloc[0]
                df["_close"] = _close


                df["_open"] = df["_close"].shift(1)
                _k = df.columns.get_loc("_open")
                k = df.columns.get_loc("open")
                df.iloc[0,_k] = df.iloc[0,k]

                df["_high"] = df[["_open","_close"]].max(axis=1) + _high
                df["_low"]  = df[["_open","_close"]].min(axis=1) - _low

                df["_volume"] = _volume


                frame = {
                    "date":time,
                    "open":df["_open"],
                    "high":df["_high"],
                    "low":df["_low"],
                    "close":df["_close"],
                    "volume":df["_volume"],
                    "close_time":close_time
                }

                returnDF = pd.DataFrame(frame)

                
                # Generate the data untill there are no negative values
                if not (returnDF.iloc[:,1:4] < 0).any().any():
                    _continue = False

            else:
                print("Error, wrong method number provided!")
                
                _continue = False
        
        return returnDF

class GenerateCandles(Noise):
    """
    This class is for making random stock candlestick charts with different methods
    """

    def __init__(self) -> None:
        
        pass

    def BrownianMotion(self, candlesCount:int, volatility:float, wicks:list, maxMargin, mean = 0, startDate = "2022-01-01", frequency = "5min") -> pd.DataFrame:
        """
        Generates a brownian motion (Also know ans wiener process) with the given parameters.
        Note that the brownian motion is used to generate random stock prices as a time series 
        representable in a line chart. But in some startegies we need candlestick data (open, 
        high, low, close) to be able to run the backtest. We have chosen the random variables 
        generated by wiener process as the closing price of each candle. Also the oppening of
        the next candle is the closing of the previous candle. The only challange is to generate
        the high and low of the candle. we have used a normal distribution to generate theese 
        values but the maximum limit for the shadows will be a percentage of the cnalde body.
        This value will be a parameter of the function.
        
        Updates:
        --------
        1. Some charting applications require indexes to be date, so to make the data more suitable 
        for these applications (i.e. mplfinance) we add a date column as well to the outputted 
        dataframe. The starting date of the data is set to 2022-01-01 as a default but it can be 
        changed by the user.

        2. The candlestick data has to be positive, but the wiener process generates random numbers
        and these random numbers can be negative. So we have to make sure that the candlestick
        dataset is positive. To do this, we generate a random number between 1 and "maxMargin"
        and shift the chart (If it has negative prices) to the generated number.



        Parameters
        ----------
        candlesCount: int: number of candles to generate
        mean: float: mean of the brownian motion, The default is 0
        volatility: float: variance of the brownian motion
        wicks: list[lower, upper]: the length of the upper and lower wicks as a percentage of the
            candle body The values has to be in percentage(e.g. 50 for 50%)
        startDate: str: starting date of the data (optional)
        frequency: str: frequency of the data (optional)
        
        To do:
        ------
        * Adding the volume for each candle (The volume will be proportional to the candle's body).
        Currently the volume is set to 0.

        Returns
        -------
        df: pandas.DataFrame: The generated candles containing the date, open, high, low, close, 
        close_time and volume. (date is the oppening time of the candle)
        """

        # Because of shifting, we lose the last candle, so we add one to the candlesCount
        candlesCount += 1
        wicks = [n/100 for n in wicks]

        # The dataframe to be returned
        df = pd.DataFrame()
        
        random = np.random.normal(loc = mean, scale = 1 * volatility, size = candlesCount)

        df["date"] = pd.date_range(start = startDate, periods = candlesCount, freq = "5min")
        df["open"] = random.cumsum()
        df["close"] = df["open"].shift(-1)

        _body = df["close"] - df["open"]
        
        # The amount of nose to for generating the high and low wicks
        rand1 = np.random.normal(loc = 0, scale = 1 , size = candlesCount); rand1 = rand1/abs(rand1).max()
        rand2 = np.random.normal(loc = 0, scale = 1 , size = candlesCount); rand2 = rand2/abs(rand2).max()
        
        up = df[["open","close"]].max(axis=1)
        down = df[["open","close"]].min(axis=1)

        df["low"] = down - abs(_body) * (1 + wicks[0]) * rand2
        df["high"] = up + abs(_body) * (1 + wicks[1]) * rand1
        df["volume"] = 0
        df["close_time"] = df["date"].shift(-1)

        # Shifting the data to make negative prices positive
        lowestNum = df["low"].min()

        if lowestNum < 0:
            shift = randint(1,maxMargin)

            df["open"] += shift -lowestNum
            df["close"] += shift -lowestNum
            df["low"] += shift -lowestNum
            df["high"] += shift -lowestNum

        return df.iloc[:-2,:]

    def GeometricBrownianMotion(self,
        method,n) -> pd.DataFrame():
        """
        Generates valid synthetic stock candlesticks with geometric brownian motion.
        Two methods are abalible for feeding the random data: 1.Random walk and 2.Normal distribution

        
        Arguments
        ---------
            method: string: Two options possible, "RandomWalk" and "NormalDist" which defime the ways
            that the stochastic process continues. Both are calculated through wiener process
            n: int: Number of canldes to generate
        
        Returns
        -------
            A pd.Dataframe() with columns = ["Date", "open", "high", "low", "close"]
        """
        
        if method == "RandomWalk":
            for i in range(0,n):

                pass
            
        elif method == "NormalDist":
            pass




cls = GenerateCandles()

candles = cls.BrownianMotion(candlesCount = 100000, mean = 0, maxMargin = 50000, volatility = 500, wicks = [50, 50], frequency = "5m")

# saveCandles(candles, location = "Candlestick-Noise/Data/Generated-1year-50,50-5min-vola-i.pkl", saveImage = True, method = "plotly")

plotDataset(candles, method = "mplfinance")


# # The code to add noise to candles
      
# # data = getCandles("./Datas/historical_BTCUSDT_5m_1420057800000.csv")[-1000:]

# data = pd.read_json("./Candlestick-Noise/Data/data.json")
# data = data["2021-01-01" <= data["date"]]
# data = data[data["date"] <= "2022-01-01"]

# originalChart = go.Candlestick(
#                 x=data["date"][0:1000],
#                 open=data['open'][0:1000],
#                 high=data['high'][0:1000],
#                 low=data['low'][0:1000],
#                 close=data['close'][0:1000],
#                 name = "Original"
#                 )

# candles = Noise().GaussianCnadles(
#     data["date"], 
#     data["open"], 
#     data["high"], 
#     data["close"], 
#     data["low"], 
#     data["volume"], 
#     data["close_time"], 
#     scale = [50, 0, 0], 
#     method = 2
#     )

# # candles.to_pickle("./Python 18-Generating different data/Candlestick-Noise/Data/generated_2021_01_01__50,0,0_3.pkl")
# # candles = pd.read_pickle("Candlestick-Noise\Data\generated_2021_01_01___2022_01_01__100,20,20_3.pkl")

# noisedChart = go.Candlestick(
#                 x=candles["date"][0:1000],
#                 open=candles['open'][0:1000],
#                 high=candles['high'][0:1000],
#                 low=candles['low'][0:1000],
#                 close=candles['close'][0:1000],
#                 name = "Noise",
#                 increasing={'line': {'color': 'blue'}},
#                 decreasing={'line': {'color': 'purple'}},
#                 )

# fig = go.Figure(data=[noisedChart, originalChart])

# fig.show()
