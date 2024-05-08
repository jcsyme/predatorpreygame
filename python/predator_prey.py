from typing import *
import climate as cl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import support_functions as sf
import warnings


class Strategy:
    """
    Simple class for storing strategy information
    """

    def __init__(self,
        index: int,
        function: Callable,
        name: str,
    ) -> None:

        self._initialize_properties(
            index,
            function,
            name,
        )

        return None

    

    def __call__(self,
        *args,
        **kwargs,
    ) -> Any:

        if self.function is None:
            return None

        # otherwise, return function value    
        out = self.function(
            *args,
            **kwargs,
        )

        return out



    def _initialize_properties(self,
        index: int,
        function: Callable,
        name: str,
    ) -> None:
        """
        Initialize elements of the strategy
        """

        function = function if isinstance(function, Callable) else None
        index = index if sf.isnumber(index, integer = True) else None
        name = str(name)


        ##  SET PROPERTIES

        self.function = function
        self.index = index
        self.name = name

        return None



class PredatorPreyModel:
    """
    Build a model of predator-prey interactions. Allows for the projection of 
        predator and prey populations based on INFOHERE

    PARAMETERS:
        - a: prey growth parameter
        - b: inverse of predator growth parameter (1/predator growh)
        - b_min: minmium allowable value of b when subject to climate change. 
            Default is 0.25
        - gamma: exponential dependence of b on T
        - n_time_periods: number of time periods to project for
        - t: time
        - T: temperature

    For information on climate parameters (including C, F, F0, and S), use
        ?ClimateChangeModel (accessible using ?PredatorPreyModel.climate_model)

    
    Initialization Arguments
    ------------------------
    - predator_0: initial population of predators (must be >= 0)
    - prey_0: initial population of predators (must be >= 0)
    - n_time_periods: default number of time periods to run for
    
    Optional Arguments
    ------------------
    - C: climate parameter C, >= 0. If None, defaults to ClimateChangeModel
        defaults
    - F0: initial forcing rate, >= 0. If None, defaults to ClimateChangeModel
        defaults
    - b_min: Default of 0.25. Minmium allowable value of b when subject to 
        climate change. 
    - dict_strategies: optinoal dictionary mapping strategy names to a function, 
        where each funciton has the following arguments:

            function_name_here(
                vec_prey,
                vec_harvest,
                temperature,
                t,
            )

            EXAMPLE: 

                an example dictionary would be 

                dict_strategies = {
                    "soft_constant": strategy_soft_constant,
                    "other_strat": strategy_other,
                }

    - discount_rate: discount rate to use for scoring harvests to calculate the
        net present value
    - field_a: field storing the value of the parameter a
    - field_appendage_climate_change: appendage to other fields to denote
        context is dependent on climate change
    - field_appendage_static: appendage to other fields to denote
        context is independent of climate change (i.e., static)
    - field_b: field storing the value of the parameter b
    - field_harvest: field storing harvests
    - field_npv: field storing the net present value
    - field_population_predator: generic field storing the population of 
        predator
    - field_population_predator: generic field storing the population of 
        prey
    - field_regret: field storing the regret
    - field_temperature: field to use to denote temperature
    - field_time_period: field to use for time periods
    - gamma: Default of 0.3. Hyperparameter that affects the value of b under 
        climate change (exponential dependence of b on T).
    """
    def __init__(self,
        predator_0: Union[float, int],
        prey_0: Union[float, int],
        n_time_periods: int,
        C: Union[float, int, None] = None,
        F0: Union[float, int, None] = None,
        b_min: Union[float, int, None] = None,
        dict_strategies: Union[Dict[str, Tuple], None] = None,
        discount_rate: float = 0.05,
        gamma: Union[float, int, None] = None,
        **kwargs,
    ) -> None:
        
        self._initialize_parameters(
            predator_0,
            prey_0,
            n_time_periods,
            b_min = b_min,
            discount_rate = discount_rate,
            gamma = gamma,
        )

        self._initialize_climate(
            C = C,
            F0 = F0,
        )

        self._initialize_fields(**kwargs, )
        self._initialize_properties()
        self._initialize_strategies(
            dict_strategies = dict_strategies,
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
    
    def _initialize_climate(self,
        C: Union[float, int, None] = None,
        F0: Union[float, int, None] = None,
    ) -> None:
        """
        Set the following climate-related parameters: 

            * self.climate_model


        Function Arguments
        ------------------
        
        Keyword Arguments
        -----------------
        - C: climate parameter C, >= 0. If None, defaults to ClimateChangeModel
            defaults
        - F0: initial forcing rate, >= 0. If None, defaults to 
            ClimateChangeModel defaults
        """

        C = max(C, 0) if sf.isnumber(C) else C
        F0 = max(F0, 0) if sf.isnumber(F0) else F0

        climate_model = cl.ClimateChangeModel(
            C = C, 
            F0 = F0,
            n_time_periods = self.n_time_periods,
        )

        
        ##  SET PROPERTIES

        self.climate_model = climate_model

        return None



    def _initialize_fields(self,
        field_a: str = "a",
        field_appendage_climate_change: str = "climate_change",
        field_appendage_static: str = "static",
        field_b: str = "b",
        field_harvest: str = "harvest",
        field_npv: str = "net_present_value",
        field_population_predator: str = "population_predator",
        field_population_prey: str = "population_prey",
        field_regret: str = "regret",
        field_temperature: str = "temperature",
        field_time_period: str = "time_period",
    ) -> None:
        """
        Initialize fields used in output data frames and schema. Sets the 
            following properties:

            * self.field_a
            * self.field_appendage_climate_change
            * self.field_appendage_static
            * self.field_b
            * self.field_npv
            * self.field_population_predator_climate_change:
                field storing the population of predators WITH climate change
            * self.field_population_prey_climate_change
                field storing the prey of predators WITH climate change
            * self.field_population_predator_static:
                field storing the population of predators WITHOUT climate change
                (i.e., under static assumptions)
            * self.field_population_prey_static:
                field storing the population of prey WITHOUT climate change
                (i.e., under static assumptions)
            * self.field_population_predator
            * self.field_population_prey
            * self.field_regret
            * self.field_temperature
            * self.field_time_period


        Function Arguments
        ------------------
        
        Keyword Arguments
        -----------------
        - field_a: field storing the value of the parameter a
        - field_appendage_climate_change: appendage to other fields to denote
            context is dependent on climate change
        - field_appendage_static: appendage to other fields to denote
            context is independent of climate change (i.e., static)
        - field_b: field storing the value of the parameter b
        - field_harvest: field storing harvests
        - field_npv: field storing the net present value
        - field_population_predator: generic field storing the population of 
            predator
        - field_population_predator: generic field storing the population of 
            prey
        - field_regret: field storing the regret
        - field_temperature: field to use to denote temperature
        - field_time_period: field to use for time periods
        """ 

        ##  COULD IMPLEMENT CHECKS HERE

        field_population_predator_climate_change = f"{field_population_predator}_{field_appendage_climate_change}"
        field_population_prey_climate_change = f"{field_population_prey}_{field_appendage_climate_change}"
        field_population_predator_static = f"{field_population_predator}_{field_appendage_static}"
        field_population_prey_static = f"{field_population_prey}_{field_appendage_static}"


        ##  SET PROPERTIES

        self.field_a = field_a
        self.field_appendage_climate_change = field_appendage_climate_change
        self.field_appendage_static = field_appendage_static
        self.field_b = field_b
        self.field_harvest = field_harvest
        self.field_npv = field_npv
        self.field_population_predator_climate_change = field_population_predator_climate_change
        self.field_population_prey_climate_change = field_population_prey_climate_change
        self.field_population_predator_static = field_population_predator_static
        self.field_population_prey_static = field_population_prey_static
        self.field_population_predator = field_population_predator
        self.field_population_prey = field_population_prey
        self.field_regret = field_regret
        self.field_temperature = field_temperature
        self.field_time_period = field_time_period

        return None



    def _initialize_parameters(self,
        predator_0: Union[float, int],
        prey_0: Union[float, int],
        n_time_periods: int,
        b_min: Union[float, int, None] = None,
        discount_rate: Union[float, None] = None,
        gamma: Union[float, int, None] = None,
    ) -> None:
        """
        Initialize model parameters for the PredatorPreyModel. Sets the 
            following properties:
            
            * self.b_min
            * self.discount_rate
            * self.gamma
            * self.n_time_periods:
                number of time periods to project (by default)
            * self.predator_0:
                initial population of predator
            * self.prey_0:
                initial population of prey

        Function Arguments
        ------------------
        - predator_0: initial population of predators (must be >= 0)
        - prey_0: initial population of predators (must be >= 0)
        - n_time_periods: number of time periods to run for. Defaults to 100 if 
            set incorrectly.
        
        Keyword Arguments
        -----------------
        - b_min: minmium allowable value of b when subject to climate change. 
            Default is 0.25
        - discount_rate: discount rate to use for scoring harvests to calculate 
            the net present value
        - gamma: hyperparameter that affects the value of b under climate change
            (exponential dependence of b on T)
        """ 

        # verify time periods
        n_time_periods = (
            100 
            if not isinstance(n_time_periods, int) 
            else max(n_time_periods, 1)
        )
        
        # verify predator/prey populations
        predator_0 = 0 if not sf.isnumber(predator_0) else max(predator_0, 0)
        prey_0 = 0 if not sf.isnumber(prey_0) else max(prey_0, 0)
        
        # verify some other parameters
        b_min = 0.25 if not sf.isnumber(b_min) else max(b_min, 0.0)
        gamma = 0.3 if not sf.isnumber(gamma) else max(gamma, 0)
        
        discount_rate = 0.05 if not sf.isnumber(discount_rate) else discount_rate


        ##  SET PROPERTIES
        
        self.b_min = b_min
        self.discount_rate = discount_rate
        self.gamma = gamma
        self.n_time_periods = n_time_periods
        self.predator_0 = predator_0
        self.prey_0 = prey_0
        
        return None
    


    def _initialize_properties(self,
    ) -> None:
        """
        Initialize other properties of the PredatorPreyModel. Sets the following
            properties:

            * self.df_projected_last:
                most recent projected data frame
            * self.dict_metrics_last:
                more recent metrics associated with the run
            * self.strategy_function_arguments_ordered:
                ordered argument requirements for the strategy function


        Function Arguments
        ------------------
        
        Keyword Arguments
        -----------------
        """ 

        strategy_function_arguments_ordered = [
            "vec_prey",
            "vec_harvest",
            "temperature",
            "t"
        ]


        ##  SET PROPERTIES

        self.df_projected_last = None
        self.dict_metrics_last = None
        self.strategy_function_arguments_ordered = strategy_function_arguments_ordered

        return None
    


    def _initialize_strategies(self,
        dict_strategies: Union[Dict[int, Tuple], None ] = None,
    ) -> None:
        """
        Set the following climate-related parameters: 

            * self.


        Function Arguments
        ------------------
        
        Keyword Arguments
        -----------------
        - dict_strategies: dictionary mapping strategy names to a function with
            the following arguments

            function_name_here(
                vec_prey,
                vec_harvest,
                temperature,
                t,
            )

            EXAMPLE: 

                an example dictionary would be 

                dict_strategies = {
                    "soft_constant": strategy_soft_constant,
                    "other_strat": strategy_other,
                }
        """

        # space of values
        self.all_strategy_names = []
        self.all_strategy_indices = []
        
        # some dictionaries
        self.dict_strategies = {}
        self.dict_strategy_name_to_indices = {}
        self.dict_strategy_index_to_name = {}

        # initialize dictionary and spaces
        dict_strategies_init = {
            "baseline": self.constant_harvest
        }
        self._add_strategies(dict_strategies_init, ) 
        self._add_strategies(dict_strategies, ) 

        return None

    


    ############################
    #    NON-INIT FUNCTIONS    #
    ############################

    def _add_strategies(self,
        dict_strategies_to_functions: Dict[str, Tuple],
    ) -> None:
        """
        Add strategies in dict_strategies_to_functions to the model
        
        Function Arguments
        ------------------
        - dict_strategies: dictionary mapping strategy names to a function with
            the following arguments

            function_name_here(
                vec_prey,
                vec_harvest,
                temperature,
                t,
            )

            EXAMPLE: 

                an example dictionary would be 

                dict_strategies = {
                    "soft_constant": strategy_soft_constant,
                    "other_strat": strategy_other,
                }
        """

        # initialize the index
        ind = (
            max(self.all_strategy_indices) + 1
            if len(self.all_strategy_indices) > 0
            else 0
        )

        
        if not isinstance(dict_strategies_to_functions, dict):
            return None

        # iterate
        for k, v in dict_strategies_to_functions.items():

            nm = str(k)

            # check the function is valid; if not, skip
            try:
                self.verify_strategy_function(
                    v, 
                    stop_on_error = True, 
                )
            
            except Exception as e:
                continue
            
            strat = Strategy(
                ind,
                v,
                nm
            )

            self.all_strategy_indices.append(ind)
            self.all_strategy_names.append(nm)
            self.dict_strategy_index_to_name.update({ind: nm})
            self.dict_strategy_name_to_indices.update({nm: ind})
            self.dict_strategies.update({nm: strat})

            ind += 1

        return None
    


    def get_strategy(self,
        ind_specification: Union[str, int],
    ) -> Union[Strategy, None]:
        """
        Return a strategy based on a name (string) or index (integer)
        """

        if isinstance(ind_specification, int):
            ind_specification = self.dict_strategy_index_to_name.get(ind_specification)
            if ind_specification is None:
                return None

        if not isinstance(ind_specification, str):
            return None
        
        out = self.dict_strategies.get(ind_specification)

        return out
    


    def verify_strategy_function(self,
        z: callable,
        stop_on_error: bool = False,
    ) -> Union[callable, None]:
        """
        Verify a strategy function. If the input function is invalid, returns
            the default constant harvest function (which defauls to 0 harvest)

        NOTE: the strategy function must have the following ordered arguments:

            * vec_prey: vector of prey populations (listlike)
            * vec_harvest: vector of historical harvest (listlike)
            * temp: temperature (number)
            * t: time period (integer >= 0)

        Function Arguments
        ------------------
        - z: function to check

        Keyword Arguments
        -----------------
        - stop_on_error: throw an error if the function is invalid?
        """

        
        # if not callable, return the constant harvest function
        if not callable(z):

            if stop_on_error:
                tp = str(type(z))
                msg = f"""
                Error in verify_strategy_function(): invalid type '{tp}'
                specified for z. Must be a Callable.
                """
                raise RuntimeError(msg)

            return self.constant_harvest
        
        # get arguments and check
        args, kwargs = sf.get_args(z)

        acceptable = len(args) >= 4
        acceptable &= (
            args[0:4] == self.strategy_function_arguments_ordered
            if acceptable
            else False
        )

        func_out = z if acceptable else self.constant_harvest

        return func_out

    


    ############################
    #    CORE FUNCTIONALITY    #
    ############################

    def get_climate_b(self,
        b0: Union[float, int],
        F: Union[float, int], 
        S: Union[float, int, np.ndarray], 
        b_min: Union[float, int, None] = None,
        gamma: Union[float, int, None] = None,
        **kwargs,
    ) -> Union[np.ndarray, None]:
        """
        Retrieve the vector of b as a function of temperature. Calls 
            self.climate_model.project_temperature_change() to project temperature.
            Returns None if b0 is misspecified.

        Function Arguments
        ------------------
        - F: radiative forcing parameter
        - S: climate sensitiviy

        Keyword Arguments
        -----------------
        - b_min: minimum value of b
        **kwargs: passed to ClimateChangeModel.project_temperature_change(). Valid
            arguments are:
            * C: climate parameter C
            * F0: initial forcing rate
            * n_time_periods: default number of time periods to run for
        """

        if not (sf.isnumber(F) & sf.isnumber(S)):
            return None

        # get the temperature vector
        vec_temp = self.climate_model.project_temperature_change(
            F,
            S,
            **kwargs
        )

        # get some parameters
        b_min = self.get_b_min(b_min)
        gamma = self.get_gamma(gamma)

        b0 = b0[0] if sf.islistlike(b0) else b0
        if not sf.isnumber(b0):
            return None
        
        # Under climate change, b (predator growth) increases with increasing
        # temperature according to b(t) = b0 e^(-gamma * T(t))
        vec_b = sf.vec_bounds(b0*np.exp(-gamma * vec_temp), (b_min, np.inf))

        return vec_b
    


    def get_gamma(self,
        gamma: Union[float, int, None] = None,
    ) -> int:
        """
        Get the hyperparameter gamma (exponential dependence of b on T)
        """
        gamma = self.gamma if not sf.isnumber(gamma) else max(gamma, 1)

        return gamma
    


    def get_b_min(self,
        b_min: Union[float, int, None] = None,
    ) -> int:
        """
        Get the minimum value of b
        """
        b_min = self.b_min if not sf.isnumber(b_min) else max(b_min, 0.0)

        return b_min



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
    


    def get_predator_init(self,
        predator_0: Union[float, int, None] = None,
    ) -> int:
        """
        Get initial predator population from a function argument
        """

        predator_0 = (
            self.predator_0 
            if not sf.isnumber(predator_0) 
            else max(predator_0, 0)
        )

        return predator_0



    def get_prey_init(self,
        prey_0: Union[float, int, None] = None,
    ) -> int:
        """
        Get initial prey population from a function argument
        """

        prey_0 = (
            self.prey_0 
            if not sf.isnumber(prey_0) 
            else max(prey_0, 0)
        )

        return prey_0
    


    def get_vec_b(self,
        b: Union[float, int, np.ndarray], 
        include_climate: Union[bool, None] = None,
        n_time_periods: Union[int, None] = None,
        **kwargs,
    ) -> Union[np.ndarray, None]:
        """
        Retrieve the vector b for use in the predator-prey model based on input
            conditions (e.g., is climate factored in?)

        Function Arguments
		------------------
		- b: parameter b (can optionally be passed as a vector)
            NOTE: if include_climate == True, then b should be passed only as
            a scalar. If b is passed as a vector and include_climate == True,
            then b[0] will be used as the scalar.
        - vec_temperature: vector of 

		Keyword Arguments
		-----------------
        - include_climate: include climate change impact on temperature?
        - n_time_periods: optional number of time periods to project for
        **kwargs: passed to self.get_climate_b(),
        """
        
        if include_climate:
            
            # get b
            b = b if sf.isnumber(b) else (b[0] if sf.islistlike(b) else None)
            if b is None:
                return None

            # strip kwargs
            F = kwargs.get("F")
            S = kwargs.get("S")
            kwargs_pass = dict(
                (k, v) for k, v in kwargs.items() if k not in ["F", "S"]
            )
            kwargs_pass.update({"n_time_periods": n_time_periods})

            # if F or S are not specified, will return None
            vec_b = self.get_climate_b(b, F, S, **kwargs_pass)

        else:

            n_time_periods = self.get_n_time_periods(n_time_periods)

            vec_b = (
                np.full(n_time_periods + 1, b)
                if sf.isnumber(b)
                else (
                    np.ndarray(b)
                    if sf.islistlike(b)
                    else None
                )
            )

        return vec_b


    
    def calculate_metrics(self,
        df_projection: pd.DataFrame,
        discount_rate: Union[int, float] = 0.05,
    ) -> Union[dict, None]:
        """
        Calculate metrics associated with the projection. Returns the Net 
            Present Value (NPV) and Regret of a run in dictionary with keys 
            self.field_npv and self.field_regret.

        NOTE: 
            * NPV is the net present value of the harvest, summed over all years
                and discounted using `discount rate`. 
            * Regret is the total percentage of all prey that was not harvested
            * If the prey is driven to extinction at any point during the 
                simulation, then NPV = -1 and regret = 1

        Function Arguments
        ------------------
        - df_projection: projection output data frame 
        - discount_rate: discount rate to use for NPV calculation. Default is 5%
        """

        ##  SOME CHECKS

        set_req_fields = set({
            self.field_harvest,
            self.field_population_prey,
            self.field_time_period,
        })
        return_none = not isinstance(df_projection, pd.DataFrame)
        return_none |= (
            not set_req_fields.issubset(set(df_projection.columns))
            if not return_none
            else return_none
        )

        if return_none:
            return None

        # get vectors and calculate

        vec_harvest = df_projection[self.field_harvest].to_numpy()
        vec_prey = df_projection[self.field_population_prey].to_numpy()
        vec_time_period = df_projection[self.field_time_period].to_numpy()

        # calculate NPV and Regret metrics
        npv = (
            (vec_harvest/((1 + discount_rate)**vec_time_period)).sum()
            if vec_prey.min() != 0.0
            else -1.0
        )
        
        regret = (
            (vec_prey - vec_harvest).sum()/vec_prey.sum()
            if vec_prey.min() != 0.0
            else 1.0
        )

        # build the output dictionary and return it
        dict_out = {
            self.field_npv: npv,
            self.field_regret: regret
        }

        return dict_out



    def constant_harvest(self,
        *args,
        return_value: Union[float, int] = 0.0,
    ) -> int:
        """
        Dummy function to replace invalid specifications of a harvest; returns
            `return_value` no matter the input parameters.
        """
        
        return return_value
    


    def get_project_return(self,
        tup: Tuple[pd.DataFrame, dict],
        metrics_only: bool,
    ) -> Union[Tuple[pd.DataFrame, dict], pd.DataFrame]:
        """
        Support for project(). Determine and format return
        """

        out = tup

        if metrics_only:

            out = tup[1]

            out = dict((k, [v]) for k, v in out.items())
            out = pd.DataFrame(out)

        return out
    


    def plot_outcomes(self,
        df_projection: Union[pd.DataFrame, None] = None,
        dict_metrics: Union[dict, None] = None,
        figsize: Tuple[int, int] = (15, 10),
        ax: Union[np.ndarray, None] = None,
        **kwargs,
    ) -> plt.plot:
        """
        Plot the population of prey, predators, and harvest from a model 
            projection.
        
        Function Arguments
        ------------------
        
        Keyword Arguments
        -----------------
        - df_projection: dataframe produced by PredatorPrey.project(). If None,
            defaults to self.df_projected_last (outputs from the last model run)
        - dict_metrics: scoring dictionary produced by PredatorPrey.project(). 
            If None, defautls to self.dict_metrics_last (outputs from the last
            model run)
        - figsize: default figure size to use
        - ax: optional axis to pass
        """
        
        ##  INITIALIZE
        
        # check inputs
        if not isinstance(df_projection, pd.DataFrame):
            df_projection = self.df_projected_last
            dict_metrics = self.dict_metrics_last

        # get metrics if they are missing 
        if not isinstance(dict_metrics, dict):
            dict_metrics = self.calculate_metrics(df_projection)
        
        # get fields to plot
        fields = [
            self.field_time_period, 
            self.field_harvest,
            self.field_population_predator,
            self.field_population_prey
        ]


        ##  PLOT OUTCOMES

        # setup the figure -- check figure size specification
        if isinstance(figsize, tuple):
            accept_fs = len(figsize) == 2
            accept_fs = (
                sf.isnumber(figsize[0], integer = True) & sf.isnumber(figsize[1], integer = True)
                if accept_fs
                else False
            )

        if ax is None:
            figsize = (15, 10) if not accept_fs else figsize
            fig, ax = plt.subplots(figsize = figsize, )
        
        # get metrics
        npv = np.round(dict_metrics.get(self.field_npv), decimals = 4)
        regret = np.round(dict_metrics.get(self.field_regret), decimals = 4)

        
        plot = (
            df_projection[fields]
            .plot(
                x = self.field_time_period,
                ax = ax,
                linewidth = 3,
                title = f"Predator/Prey Populations and Harvest\n\nNPV = {npv}, Regret = {regret}",
                **kwargs,
            )
        )
        
        return plot
    


    def project(self,
        a: Union[float, int], 
        b: Union[float, int, np.ndarray], 
        F: Union[float, int, None] = None,
        S: Union[float, int, None] = None,
        discount_rate: Union[float, None] = None,
        include_climate: Union[bool, None] = None,
        metrics_only: bool = False,
        n_time_periods: Union[int, None] = None,
        predator_0: Union[float, int, None] = None,
        prey_0: Union[float, int, None] = None,
        verify_strategy: bool = True,
        z: Union[callable, None] = None, 
        **kwargs,
    ) -> Union[Tuple[pd.DataFrame, dict], pd.DataFrame, None]:
        """
        Project the predator/prey model

        Function Arguments
		------------------
		- a: parameter a
        - b: parameter b (can optionally be passed as a vector)

		Keyword Arguments
		-----------------
        - F: parameter F for radiative forcing, required if 
            include_climate == True
        - S: parameter S for climate sensitivity, required if 
            include_climate == True
        - discount_rate: optional specification of a discount rate for 
            calculating the Net Present Value. If invalid, defaults to 
            self.discount_rate
        - include_climate: include climate change impact on temperature? Must 
            specify F and S to successfully pass this. 
            * If None, will assume True if both F and S are passed. Otherwise,
                can set to False to spcifically avoid passing climate
        - metrics_only: return only the metrics associated with a run? If True,
            returns a data frame with metrics.
        - n_time_periods: optional number of time periods to project for
        - predator_0: optional specification of initial predator population. If
            None, defaults to self.predator_0
        - prey_0: optional specification of initial prey population. If None, 
            defaults to self.prey_0
        - verify_strategy: Verify the strategy function? can be set to False for
            batch runs
        - z: strategic harvest function. If None, defaults to no harvest
        """

        try:
            tup = self.project_func(
                a,
                b,
                F = F,
                S = S,
                discount_rate = discount_rate,
                include_climate = include_climate,
                n_time_periods = n_time_periods,
                predator_0 = predator_0,
                prey_0 = prey_0,
                verify_strategy = verify_strategy,
                z = z,
                **kwargs
            )
        
        except Exception as e:

            self.df_projected_last = None
            self.dict_metrics_last = None

            raise RuntimeError(f"Error running the Predatory/Prey model: {e}")

        # assign properties and return
        self.df_projected_last = tup[0].copy()
        self.dict_metrics_last = tup[1].copy()

        # return metrics only?
        tup = self.get_project_return(tup, metrics_only)

        return tup
    


    def project_ema(self,
        a: Union[float, int] = 0.0,
        b: Union[float, int, np.ndarray] = 0.0,
        F: Union[float, int, None] = 0.0,
        S: Union[float, int, None] = 0.0,
        strategy: int = 0, # key in dict_strategies_to_functions
        discount_rate: Union[float, None] = None, # will default to model default
    ) -> pd.DataFrame:
        """
        Wrapper function to support analysis in EMA workbench. The keyword arguments are those that can be explored over using EMA.
        """

        #
        z = self.get_strategy(strategy, )
        if z is None:
            raise RuntimeError(f"Invalid strategy {strategy}: strategy not found in PredatoryPreyModel")
        
        # project and get metrics
        df_projection, dict_metrics = self.project(
            a,
            b,
            F = F,
            S = S,
            discount_rate = discount_rate,
            verify_strategy = False,
            z = z.function,
        )

        # EMA reads outputs as a tuple
        npv = dict_metrics.get(self.field_npv)
        regret = dict_metrics.get(self.field_regret)
        tup_out = (npv, regret)

        return tup_out    




    def project_func(self,
        a: Union[float, int], 
        b: Union[float, int, np.ndarray], 
        F: Union[float, int, None] = None,
        S: Union[float, int, None] = None,
        discount_rate: Union[float, None] = None,
        include_climate: Union[bool, None] = None,
        n_time_periods: Union[int, None] = None,
        predator_0: Union[float, int, None] = None,
        prey_0: Union[float, int, None] = None,
        verify_strategy: bool = True,
        z: Union[callable, None] = None, 
        **kwargs,
    ) -> Union[Tuple[pd.DataFrame, dict], pd.DataFrame, None]:
        """
        Project the predator/prey model. Base function (wrapped by project())

        Function Arguments
		------------------
		- a: parameter a
        - b: parameter b (can optionally be passed as a vector)

		Keyword Arguments
		-----------------
        - F: parameter F for radiative forcing, required if 
            include_climate == True
        - S: parameter S for climate sensitivity, required if 
            include_climate == True
        - discount_rate: optional specification of a discount rate for 
            calculating the Net Present Value. If invalid, defaults to 
            self.discount_rate
        - include_climate: include climate change impact on temperature? Must 
            specify F and S to successfully pass this. 
            * If None, will assume True if both F and S are passed. Otherwise,
                can set to False to spcifically avoid passing climate
        - n_time_periods: optional number of time periods to project for
        - predator_0: optional specification of initial predator population. If
            None, defaults to self.predator_0
        - prey_0: optional specification of initial prey population. If None, 
            defaults to self.prey_0
        - verify_strategy: Verify the strategy function? can be set to False for
            batch runs
        - z: strategic harvest function. If None, defaults to no harvest
        """
        ##  CHECKS
        
        # include climate? if not bool, assume that it's dependent on F & S
        if not isinstance(include_climate, bool):
            include_climate = True

        include_climate &= (sf.isnumber(F) & sf.isnumber(S))
        
        # get some key values from function arguments
        n_time_periods = self.get_n_time_periods(n_time_periods)
        predator_0 = self.get_predator_init(predator_0)
        prey_0 = self.get_prey_init(prey_0)
        
        # verify the strategy is set properly if desired
        z = self.verify_strategy_function(z) if verify_strategy else z


        ##  INITIALIZATION

        # get the discount rate for scoring NPV
        discount_rate = (
            self.discount_rate
            if not sf.isnumber(discount_rate)
            else discount_rate
        )

        vec_b = self.get_vec_b(
            b,
            F = F,
            S = S,
            include_climate = include_climate,
            n_time_periods = n_time_periods,
        )
        if vec_b is None:
            return None

        # if vec_b is valid, build output vectors
        vec_harvest = np.zeros(n_time_periods + 1)
        vec_predator = np.zeros(n_time_periods + 1)
        vec_prey = np.zeros(n_time_periods + 1)
        vec_temp = np.zeros(n_time_periods + 1)
        vec_time = np.arange(n_time_periods + 1).astype(int)
        
        # set initial states
        vec_predator[0] = predator_0
        vec_prey[0] = prey_0
        
        # initialize harvest
        h = z(
            vec_prey[0:1], # past information on prey pop
            [0], 
            vec_temp[0], # delta t is 0 in the baseline 
            0, # time period is 0
        )

        # iterate for each time period
        for t in range(n_time_periods):
            
            # get the temperature and update vector
            delta_t = self.t_inv_b(vec_b[0:(t + 1)])
            vec_temp[t] = delta_t[-1]
            
            h = 0 if not sf.isnumber(h) else h
            vec_harvest[t] = h


            ##  SETUP OUTCOMES FOR NEXT STEP

            # prey at t + 1  is function of harvest, prey, and predator in this time step
            prey_next = a*vec_prey[t]*(1 - vec_prey[t]) - vec_prey[t]*vec_predator[t] - h
            predator_next = vec_prey[t] * vec_predator[t] / vec_b[t]

            vec_prey[t + 1] = max(0, prey_next)
            vec_predator[t + 1] = max(0, predator_next)

            # do harvest before next round. Uses information on (sequential arguments)
            # - previous prey populations
            # - previous harvests
            # - temperature
            # - time period
            h = z(
                vec_prey[0:(t + 1)],
                vec_harvest[0:t], 
                vec_temp[t], 
                t, 
            )
            

        # add final outcomes
        delta_t = self.t_inv_b(vec_b)
        vec_temp[t + 1] = delta_t[-1]
        vec_harvest[t + 1] = h
        

        # build output data frame and return
        df_out = pd.DataFrame(
            {
                self.field_time_period: vec_time,
                self.field_b: vec_b,
                self.field_temperature: vec_temp,
                self.field_population_predator: vec_predator,
                self.field_population_prey: vec_prey,
                self.field_harvest: vec_harvest,
            }
        )

        dict_metrics = self.calculate_metrics(
            df_out, 
            discount_rate = discount_rate,
        )

        out = df_out, dict_metrics



        return out

    

    def t_inv_b(self,
        b: np.ndarray, 
        gamma: Union[float, int, None] = None,
    ) -> np.ndarray:
        """
        Return a vector of temperatures based on parameter b
        """
        gamma = self.get_gamma(gamma)
        vec_t = (np.log(b[0]) - np.log(b))/gamma

        return vec_t
