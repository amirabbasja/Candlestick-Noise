import string
from time import time
import pandas as pd
from plotly.subplots import make_subplots
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import plotly.graph_objects as go
# In this script we add noise to candles. There are different ways of
# adding noise to candles and all the different methods are in the Noise
# class.
def getCandles(fileLocation : string) -> pd.DataFrame():
    # This fuunction reads a dataframe from a csv file
    return pd.read_csv(fileLocation)

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
        if len(scale) == 1:
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
                # The logic: In thsi method we do not care about not generating gaps. so we add noise
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
                df["_low"] =  df[["_open","_close"]].max(axis=1) +_high
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
                # of the body + noise adn assign the value to the candle's close. and then we assign
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

                df["_high"] = df[["_open","_close"]].max(axis=1) +_high
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

    def GeometricBrownianMotion(self,
        method,n) -> pd.DataFrame():
        """
        Generates valid synthetic stock candlesticks with geometric brownian motion.
        Two methods are abalible for feeding the random data: 1.Random walk and 2.Normal distribution

        
        Arguments
        ---------
            method: string: Two options possible, "RandomWalk" and "NormalDist" which defime the ways that the stochastic process continues. Both are calculated through wiener process
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


class Brownian():
    """
    A Brownian motion class constructor
    """
    def __init__(self,x0=0):
        """
        Initialize the class
        """
        assert (type(x0)==float or type(x0)==int or x0 is None), "Expect a float or None for the initial value"
        
        self.x0 = float(x0)
    
    def gen_random_walk(self,n_step=100):
        """
        Generate motion by random walk
        
        Arguments
        ---------
            n_step: Number of steps
            
        Returns
        -------
            A NumPy array with `n_steps` points
        """
        # Warning about the small number of steps
        if n_step < 30:
            print("WARNING! The number of steps is small. It may not generate a good stochastic process sequence!")
        
        w = np.ones(n_step)*self.x0
        
        for i in range(1,n_step):
            # Sampling from the Normal distribution with probability 1/2
            yi = np.random.choice([1,-1])
            # Weiner process
            w[i] = w[i-1]+(yi/np.sqrt(n_step))
        
        return w
    
    def gen_normal(self,n_step=100):
        """
        Generate motion by drawing from the Normal distribution
        
        Arguments
        ---------
            n_step: Number of steps
            
        Returns
        -------
            A NumPy array with `n_steps` points
        """
        if n_step < 30:
            print("WARNING! The number of steps is small. It may not generate a good stochastic process sequence!")
        
        w = np.ones(n_step)*self.x0
        
        for i in range(1,n_step):
            # Sampling from the Normal distribution
            yi = np.random.normal()
            # Weiner process
            w[i] = w[i-1]+(yi/np.sqrt(n_step))
        
        return w
    
    def stock_price(
                    self,
                    s0=100,
                    mu=0.2,
                    sigma=0.68,
                    deltaT=52,
                    dt=0.1
                    ):
        """
        Models a stock price S(t) using the Weiner process W(t) as
        `S(t) = S(0).exp{(mu-(sigma^2/2).t)+sigma.W(t)}`
        
        Arguments
        ---------
            s0: Iniital stock price, default 100
            mu: 'Drift' of the stock (upwards or downwards), default 1
            sigma: 'Volatility' of the stock, default 1
            deltaT: The time period for which the future prices are computed, default 52 (as in 52 weeks)
            dt (optional): The granularity of the time-period, default 0.1
        
        Returns
        -------
            s: A NumPy array with the simulated stock prices over the time-period deltaT
        """
        n_step = int(deltaT/dt)
        time_vector = np.linspace(0,deltaT,num=n_step)
        # Stock variation
        stock_var = (mu-(sigma**2/2))*time_vector
        # Forcefully set the initial value to zero for the stock price simulation
        self.x0=0
        # Weiner process (calls the `gen_normal` method)
        weiner_process = sigma*self.gen_normal(n_step)
        # Add two time series, take exponent, and multiply by the initial stock price
        s = s0*(np.exp(stock_var+weiner_process))
        
        return s

       
# data = getCandles("./Datas/historical_BTCUSDT_5m_1420057800000.csv")[-1000:]

data = pd.read_json("./Candlestick-Noise/Data/data.json")
data = data["2021-01-01" <= data["date"]]
data = data[data["date"] <= "2022-01-01"]

originalChart = go.Candlestick(
                x=data["date"],
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name = "Original"
                )

# candles = GenerateCandles().GaussianCnadles(
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

# candles.to_pickle("./Python 18-Generating different data/Candlestick-Noise/Data/generated_2021_01_01__50,0,0_3.pkl")
candles = pd.read_pickle("Candlestick-Noise\Data\generated_2021_01_01___2022_01_01__100,20,20_3.pkl")

noisedChart = go.Candlestick(
                x=candles["date"],
                open=candles['open'],
                high=candles['high'],
                low=candles['low'],
                close=candles['close'],
                name = "Noise",
                increasing={'line': {'color': 'blue'}},
                decreasing={'line': {'color': 'purple'}},
                )

fig = go.Figure(data=[noisedChart, originalChart])

fig.show()
