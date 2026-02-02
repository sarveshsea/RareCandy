from abc import ABC, abstractmethod
from typing import List, Optional
from core.types import Candle, Signal, Bias

class Strategy(ABC):
    """
    Abstract Base Class for Trading Strategies.
    Enforces deterministic evaluation: Input(Candles) -> Output(Signal).
    """
    
    @abstractmethod
    def evaluate(self, 
                 symbol: str, 
                 candles_htf: List[Candle], 
                 candles_ltf: List[Candle],
                 bias: Optional[Bias] = None
                 ) -> Optional[Signal]:
        """
        Evaluate market data and return a Signal if entry conditions are met.
        Returns None if no signal.
        """
        pass
