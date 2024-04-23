from typing import *
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import support_functions as sf
import warnings




class ClimateChangeModel:
    """
    Simple model of climate change 

    
    Initialization Arguments
    ------------------------
    
    Optional Arguments
    ------------------
    - C: climate parameter C (defaults to 50)
    - F0: initial forcing rate (defaults to 1)
    - n_time_periods: default number of time periods to run for
    """
    

    def __init__(self,
        C: Union[float, int, None] = None,
        F0: Union[float, int, None] = None,
        n_time_periods: int = 100,
    ) -> None:
        
        self._initialize_parameters(
            C = C,
            F0 = F0,
            n_time_periods = n_time_periods,
        )
        
        return None
    
    
    
    def __call__(self,
        *args, 
        **kwargs,
    ) -> Union[dict, None]:
        
        # conditional arguments here
        out = self.project(*args, **kwargs)

        return out

    
    
    ########################
    #    INITIALIZATION    #
    ########################

    def _initialize_parameters(self,
        C: Union[float, int, None] = None,
        F0: Union[float, int, None] = None,
        n_time_periods: Union[int, None] = None,
    ) -> None:
        """
        Initialize model parameters for the ClimateChangeModel. Sets the 
            following properties:

            * self.C:
                climate parameter
            * self.F0:
                initial value of forcing
            * self.n_time_periods:
                number of time periods to project (by default)

        Function Arguments
        ------------------
        - C: climate parameter C
        - F0: initial forcing rate
        - n_time_periods: default number of time periods to run for
        
        Keyword Arguments
        -----------------
        """ 
        # verify time periods
        n_time_periods = (
            100 
            if not isinstance(n_time_periods, int) 
            else max(n_time_periods, 1)
        )
        
        # verify parameters
        C = max(C, 0) if sf.isnumber(C) else 50
        F0 = max(F0, 0)if sf.isnumber(F0) else 1
        

        ##  SET PROPERTIES
        
        self.C = C
        self.F0 = F0
        self.n_time_periods = n_time_periods
        
        return None
    


    def get_n_time_periods(self,
        n_time_periods: Union[int, None] = None,
    ) -> int:
        """
        Get number of time periods from a function argument
        """

        n_time_periods = (
            self.n_time_periods 
            if not isinstance(n_time_periods, int) 
            else max(n_time_periods, 1)
        )

        return n_time_periods
    


    def get_parameter_c(self,
        C: Union[float, int, None] = None,
    ) -> int:
        """
        Get parameter C
        """

        C = self.C if not sf.isnumber(C) else max(C, 0)

        return C



    def get_parameter_f0(self,
        F0: Union[float, int, None] = None,
    ) -> int:
        """
        Get parameter F0
        """

        F0 = self.F0 if not sf.isnumber(F0) else max(F0, 0)

        return F0



    ############################
    #    CORE FUNCTIONALITY    #
    ############################

    def project_temperature_change(self,
        F: Union[float, int], 
        S: Union[float, int, np.ndarray], 
        C: Union[float, int, None] = None,
        F0: Union[float, int, None] = None,
        n_time_periods: Union[int, None] = None,
    ) -> Union[pd.DataFrame, None]:
        """
        Project the change in average temperature as a function of radiative
            forcing and climate sensitivity. The equation is derived from the 
            following ODE:

            C dT/dt = F0 + F*t - T/S


        Function Arguments
        ------------------
        - F: radiative forcing parameter
        - S: climate sensitiviy

        Keyword Arguments
        -----------------
        - C: climate parameter C
        - F0: initial forcing rate
        - n_time_periods: default number of time periods to run for
        """

        # get and verify parameters
        C = self.get_parameter_c(C)
        F0 = self.get_parameter_f0(F0)
        n_time_periods = self.get_n_time_periods(n_time_periods)

        vec_t = np.arange(n_time_periods + 1)
        vec_temp = S*(F*vec_t + (F0 - F*S*C)*(1 - np.exp(-vec_t/(S*C))))
        
        return vec_temp