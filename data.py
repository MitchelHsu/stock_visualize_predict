import datetime as dt

from secret import API_KEY
from twelvedata import TDClient


class RealtimeData:
    def __init__(self, symbol):
        self.td = TDClient(apikey=API_KEY)
        self.symbol = symbol

    def get_data(self, start, end, interval):
        data = self.td.time_series(
            symbol=self.symbol,
            interval=interval,
            start_date=start,
            end_date=end,
            order='asc'
        )

        return data

    def get_historical(self, start=dt.date.today()-dt.timedelta(days=365), end=dt.date.today()):
        data = self.td.time_series(
            symbol=self.symbol,
            interval='1day',
            start_date=start,
            end_date=end,
            outputsize=5000,
            order='desc'
        ).as_pandas()

        return data

