import pandas as pd
import numpy as np


class Preprocessing:
    @staticmethod
    def get_time_features():
        """
        Generates a dictionary of lambda functions for extracting various time-related features from a pandas Timestamp or DatetimeIndex.

        Returns:
            dict: A dictionary where each key is the name of the feature and the value is a lambda function to extract that feature.

        Examples:
            >>> time_features = get_time_features()
            >>> df['year'] = df['timestamp'].apply(time_features['year'])
            >>> df['month'] = df['timestamp'].apply(time_features['month'])

        Notes:
            - The lambda functions are designed to work with either a single pandas Timestamp or a pandas DatetimeIndex.
        """

        return {
            "year": lambda x: x.year,
            "month": lambda x: x.month,
            "month_name": lambda x: x.month_name(),
            "week_number": lambda x: x.isocalendar().week,
            "weekday": lambda x: x.weekday()
            if isinstance(x, pd.Timestamp)
            else x.weekday,
            "is_weekend": lambda x: int(x.weekday() > 4)
            if isinstance(x, pd.Timestamp)
            else x.map(lambda y: int((y.weekday() > 4))),
            "hour": lambda x: x.hour,
            "minute": lambda x: x.minute,
            "quarter": lambda x: x.quarter,
            "season_type": (
                lambda x: x.map(
                    lambda y: "Winter"
                    if y.month < 3 or y.month == 12
                    else (
                        "Spring"
                        if y.month < 6
                        else ("Summer" if y.month < 9 else "Autumn")
                    )
                )
                if isinstance(x, pd.DatetimeIndex)
                else (
                    "Winter"
                    if x.month < 3 or x.month == 12
                    else (
                        "Spring"
                        if x.month < 6
                        else ("Summer" if x.month < 9 else "Autumn")
                    )
                )
            ),
            "week_of_month": lambda x: (x.day + x.replace(day=1).weekday() - 1) // 7 + 1
            if isinstance(x, pd.Timestamp)
            else [(i.day + i.replace(day=1).weekday() - 1) // 7 + 1 for i in x],
            "day_number": lambda x: x.dayofyear,
        }

    @staticmethod
    def add_time_features(data, *args, all_features=False):
        """
        Adds time-related features to a given DataFrame based on its index.

        Parameters:
            data (pd.DataFrame): The input DataFrame to which time features will be added.
            *args (str): Variable-length argument list specifying the names of time features to add.
            all_features (bool, optional): If True, adds all available time features to the DataFrame. Default is False.

        Returns:
            pd.DataFrame: A new DataFrame with added time features.

        Examples:
            >>> df = pd.DataFrame({'value': [1, 2, 3]}, index=pd.date_range('2021-01-01', periods=3))
            >>> add_time_features(df, 'year', 'month')
            >>> add_time_features(df, all_features=True)

        Notes:
            - The function uses the `get_time_features` function to determine which features can be added.
        """

        df = data.copy()
        if all_features:
            return Preprocessing.add_time_features(
                data, *Preprocessing.get_time_features().keys()
            )
        for feature in args:
            if feature in Preprocessing.get_time_features():
                df[feature] = Preprocessing.get_time_features()[feature](df.index)
        return df

    @staticmethod
    def cyclic_function(data, *column_names):
        """
        Adds cyclic features to a given DataFrame based on the specified columns.

        Parameters:
            data (pd.DataFrame): The input DataFrame to which cyclic features will be added.
            *column_names (str): Variable-length argument list specifying the names of columns for which cyclic features will be added.

        Returns:
            pd.DataFrame: A new DataFrame with added cyclic features.

        Examples:
            >>> df = pd.DataFrame({'month': [1, 2, 3, 4, 5]})
            >>> cyclic_function(df, 'month')
            >>> # Output DataFrame will have 'month_cos' and 'month_sin' columns.

        Notes:
            - The function uses trigonometric functions to create cyclic features.
            - The period for the cyclic features is determined by the number of unique values in the specified column.
            - The function will return a new DataFrame, leaving the original DataFrame unchanged.
        """

        df = data.copy()

        for column_name in column_names:
            period = len(df[column_name].unique())
            df[f"{column_name}_cos"] = np.cos((2 * np.pi * df[column_name]) / period)
            df[f"{column_name}_sin"] = np.sin((2 * np.pi * df[column_name]) / period)

        return df

    @staticmethod
    def radial_basis_function(x, width):
        """
        Computes the radial basis function for a given input array and width.

        Parameters:
            x (list or np.array): The input array containing the values for which the radial basis function is computed.
            width (float): The width parameter for the Gaussian function.

        Returns:
            np.array: An array containing the computed radial basis function values.

        Examples:
            >>> radial_basis_function([0, 0, 1, 1, 1, 0, 0], 1)
            array([0.6065, 0.8825, 1.0000, 0.8825, 0.6065, 0.3246, 0.1353])

            >>> radial_basis_function([0, 1, 0], 0.5)
            array([0.8825, 1.0000, 0.8825])

        Notes:
            - The function computes the radial basis function by finding the midpoints of the peaks in the input array `x`.
            - The radial basis function is computed using a Gaussian function centered at these midpoints.
        """

        midpoints = []
        i = 0
        while i < len(x):
            if (i == 0 or x[i - 1] == 0) and x[i] == 1:
                begin_peak = i
                while i < len(x) and x[i] == 1:
                    i += 1
                midpoints.append(begin_peak + ((i - begin_peak) // 2))
            else:
                i += 1

        y = np.array(range(len(x)))
        r = np.zeros(len(x))
        for i in range(len(midpoints)):
            r += np.exp(-1 / (2 * width**2) * (y - midpoints[i]) ** 2)

        return r

    @staticmethod
    def create_lags(data, column_name, lags):
        """
        Creates lagged columns for a given DataFrame based on the specified column and lag periods.

        Parameters:
            data (pd.DataFrame): The DataFrame containing the column to be lagged.
            column_name (str): The name of the column to create lags for.
            lags (list): A list of integers representing the lag periods.

        Returns:
            pd.DataFrame: A DataFrame with new columns for each specified lag period.

        Examples:
            >>> df = pd.DataFrame({'value': [1, 2, 3, 4, 5]})
            >>> create_lags(df, 'value', [1, 2])
            >>> # Output DataFrame will have 'value_lag_1D' and 'value_lag_2D' columns.

        Notes:
            - The function will return a new DataFrame, leaving the original DataFrame unchanged.
            - The new lagged columns will have NaN values for the initial periods equal to the lag value.
        """

        df = data.copy()
        for lag in lags:
            lag_column_name = f"{column_name}_lag_{lag}U"
            df[lag_column_name] = df[column_name].shift(lag)

        return df

    @staticmethod
    def create_windows(data, column_name, windows):
        """
        Creates rolling window columns for a given DataFrame based on the specified column and window sizes.

        Parameters:
            data (pd.DataFrame): The DataFrame containing the column to apply the rolling window to.
            column_name (str): The name of the column to create rolling windows for.
            windows (list): A list of integers representing the window sizes.

        Returns:
            pd.DataFrame: A DataFrame with new columns for each specified rolling window size.

        Examples:
            >>> df = pd.DataFrame({'value': [1, 2, 3, 4, 5]})
            >>> create_windows(df, 'value', [2, 3])
            >>> # Output DataFrame will have 'value_window_2D_mean' and 'value_window_3D_mean' columns.

        Notes:
            - The function will return a new DataFrame, leaving the original DataFrame unchanged.
            - The new rolling window columns will have NaN values for the initial periods equal to the window size minus one.
        """

        df = data.copy()
        for window in windows:
            rolling_column_name = f"{column_name}_window_{window}U_mean"
            df[rolling_column_name] = df[column_name].rolling(window=window).mean()

        return df
