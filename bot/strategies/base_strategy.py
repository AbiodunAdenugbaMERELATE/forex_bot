class BaseStrategy:
    def __init__(self, api_client, instrument, strategies):
        self.api_client = api_client
        self.instrument = instrument
        self.strategies = strategies
        self.data = []

    def on_price_update(self, data):
        raise NotImplementedError("Should implement on_price_update()")
