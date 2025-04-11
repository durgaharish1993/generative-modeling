import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from functools import reduce
import math
import os 
import json 

'''
Generating Features 
'''
class Features(object):
    """
    Class for generating features from data.
    """

    def __init__(self, CONFIG_DICT):
        """
        Initialize feature generation parameters.
        
        Args:
        - CONFIG_DICT (dict): Configuration dictionary containing parameters.
        """
        self.desc = "Generating Features"
        self.DEFAULT_WEEKS_CUTOFF = CONFIG_DICT["DEFAULT_WEEKS_CUTOFF"]
        self.BLANK_DEFAULT_WEEKS_CUTOFF = CONFIG_DICT["BLANK_DEFAULT_WEEKS_CUTOFF"]
        self.CUR_DATE = CONFIG_DICT["CUR_DATE"]
        self.NFL_END_OF_SEASON = datetime(2024,12,28,0,0) #datetime(2025,2,15,0,0)
        self.MLB_END_OF_SEASON = datetime(2024,12,28,0,0) #datetime(2024,12,31,0,0)
        self.NFL_POST_SEASON_DATE = "2025-01-25"
        self.CUTOFF_DEFAULTS = {
        "NFL": "2024-12-28",
        "MLB": "2024-12-28",
        "COLLEGE": "2024-12-28",
        "NBA": "2024-12-28",
        "OTHER": "2024-12-28",
        "NHL": "2024-12-28",
        "Buffalo Bills": self.NFL_POST_SEASON_DATE,
        "Pittsburgh Steelers": self.NFL_POST_SEASON_DATE,
        "Baltimore Ravens": self.NFL_POST_SEASON_DATE,
        "Cincinnati Bengals": self.NFL_POST_SEASON_DATE,
        "Houston Texans":self.NFL_POST_SEASON_DATE,

        "Kansas City Chiefs": self.NFL_POST_SEASON_DATE,
        "Los Angeles Chargers": self.NFL_POST_SEASON_DATE,
        "Denver Broncos": self.NFL_POST_SEASON_DATE,

        "Philadelphia Eagles": self.NFL_POST_SEASON_DATE,
        "Washington Commanders": self.NFL_POST_SEASON_DATE,

        "Detroit Lions": self.NFL_POST_SEASON_DATE,
        "Minnesota Vikings": self.NFL_POST_SEASON_DATE,
        "Green Bay Packers": self.NFL_POST_SEASON_DATE,


        "Atlanta Falcons": self.NFL_POST_SEASON_DATE,
        "Arizona Cardinals":self.NFL_POST_SEASON_DATE,
        "San Francisco 49ers":  self.NFL_POST_SEASON_DATE,
    
        "Washington Football Team": self.NFL_POST_SEASON_DATE,
        "Washington Redskins": self.NFL_POST_SEASON_DATE,
        "last_day_of_year" : self.NFL_POST_SEASON_DATE
        }

        self.DT_CUTOFF_DEFAULTS = { key : datetime.strptime(self.CUTOFF_DEFAULTS[key], '%Y-%m-%d') for key in self.CUTOFF_DEFAULTS}
        self.CUTOFF_LOGIC = "NEW"
        self.BLANK_CUTOFF_METHOD = 'OLD'
        self.PAST_PO_DEFAULT_DAYS = 14
        self.DEFAULT_DAYS_ADDITION = 14
        self.MAX_CUTOFF_LOOKBACK_DAYS = 30
        self.MAX_CUTOFF_LOOKBACK_DAYS_FOR_BLANKS  = 30 
        self.MTO_COLS = ['item_id', 'dm_sku', 'blank_ats', 'blank_total_on_po', 'blank_first_cancel_date',
                         'blank_last_cancel_date', 'blank_first_cancel_qty']

    def process_data_and_features(self, df, df_mto):
        """
        Process data and generate features.
        
        Args:
        - df (DataFrame): Main data frame.
        - df_mto (DataFrame): Data frame containing MTO data.
        
        Returns:
        - DataFrame: Processed data frame with generated features.
        """
        # Filters used
        dept_filter = df["department"].isin(["JER"])
        nfl_filter = (df['league'] == 'NFL') & (df['jersey_attribute'] == 'Game')
        mlb_filter = (df['league'] == 'MLB') & (
                (df['jersey_attribute'] == 'Replica') |
                (df['jersey_attribute'] == 'Limited')
        )

        hot_market_filter = ~df['is_hot_market']

        subset = df.loc[dept_filter]
        subset = subset.loc[(nfl_filter | mlb_filter) &  hot_market_filter]
        subset = subset.join(df_mto[self.MTO_COLS].set_index('item_id'), on='item_id')
        subset['has_dm_sku'] = ~subset['dm_sku'].isnull()
        subset['lead_time'] = subset['lead_time'].map(lambda l: -1 if l is None else int(l))
        
        # Finished PO features 
        subset["first_cancel_dt"] = pd.to_datetime(subset["first_cancel_dt"])
        subset['days_to_cancel_dt'] = subset['first_cancel_dt'].map(lambda d: (d - self.CUR_DATE).days)
        subset['has_po'] = ~subset['first_cancel_dt'].isnull()
        
        if self.CUTOFF_LOGIC == 'NEW':
            subset['cutoff_cols'] = subset.apply(lambda df_row : self.get_finished_cutoff_v2(df_row), axis=1)
        
        else:
            subset['cutoff_cols'] = subset.apply(lambda df_row: self.get_finished_cutoff_feature(df_row), axis=1)
        subset[["cutoff_dt", "raw_cutoff_in_weeks"]] = pd.DataFrame(subset['cutoff_cols'].tolist(), index=subset.index)
        subset = subset.drop(columns=["cutoff_cols"])

        # Sales level features 
        subset["last_4_wks_qty"] = subset["qty_vec"].apply(lambda x: sum(x))
        subset["last_4_wks_nd"]  = subset["nd_vec"].apply(lambda x: sum(x))
        subset["avg_dis_depth"]  = subset["disc_depth_vec"].apply(lambda x: sum(x) / len(x))
        subset["discount_bool"]  = subset["avg_dis_depth"]>0

        # Last 1 week data 
        subset["last_1_wks_qty"] = subset["qty_vec"].apply(lambda x: x[-1])
        subset["last_1_wks_nd"] = subset["nd_vec"].apply(lambda x: x[-1])

        # Fix for ats_current 
        subset.loc[subset["ats_current"] < 0, "ats_current"] = 0
        
        # Fill missing values with a default category
        subset.fillna({'derived_excl': 'none'}, inplace=True)
        
        return subset 

    '''
    Calculate Finished Inventory cutoff feature in days and weeks 
    '''
    def get_finished_cutoff_feature(self, df_row):
        """
        Calculate Finished Inventory cutoff feature in days and weeks.
        
        Args:
        - df_row (Series): Row of DataFrame containing relevant data.
        
        Returns:
        - tuple: Tuple containing cutoff days and cutoff weeks.
        """
        # Helper function to calculate cutoff weeks from days
        def calculate_cutoff_in_weeks(days):
            if days % 7 == 0:
                return int(days / 7 + 1)
            else:
                return math.ceil(days / 7)
    
        if df_row['first_cancel_dt'] is not None and df_row['days_to_cancel_dt'] > 0:
            # Calculate cutoff based on cancellation date
            in_days = df_row['days_to_cancel_dt'] + self.DEFAULT_DAYS_ADDITION
            in_weeks = calculate_cutoff_in_weeks(in_days)
            return (in_days, in_weeks)
        
        elif df_row['first_cancel_dt'] is not None and df_row['days_to_cancel_dt'] <= 0:
            # Use default cutoff days and weeks for past POs
            return (self.PAST_PO_DEFAULT_DAYS, (self.PAST_PO_DEFAULT_DAYS / 7) + 1)
        
        elif df_row['league']=='NFL' and df_row['program_group_id'] == 'REPLEN-NO':
            # Use cutoff from cur_date to 02/15 
            
            num_days          = (self.NFL_END_OF_SEASON - self.CUR_DATE).days
            in_weeks          = calculate_cutoff_in_weeks(num_days)
            return (num_days, in_weeks)
        
        elif df_row['league']=='MLB' and df_row['program_group_id'] == 'REPLEN-NO':
            # Use cutoff from cur_date to 02/15 
            num_days          = (self.MLB_END_OF_SEASON - self.CUR_DATE).days
            in_weeks          = calculate_cutoff_in_weeks(num_days)
            return (num_days, in_weeks)
    
        elif df_row['lead_time'] >= 14:
            # Calculate cutoff based on lead time
            in_days = df_row['lead_time'] + self.DEFAULT_DAYS_ADDITION 
            in_weeks = calculate_cutoff_in_weeks(in_days)
            return (in_days, in_weeks)

        else:
            # Use default cutoff days and weeks
            return (self.DEFAULT_WEEKS_CUTOFF * 7, self.DEFAULT_WEEKS_CUTOFF)


    def get_finished_cutoff_v2(self,row):

        def calculate_cutoff_in_weeks(days):
            if days % 7 == 0:
                return int(days / 7 + 1)
            else:
                return math.ceil(days / 7)
        
        # If the cutoff exists for a given team, replace last day of year with the team cutoff
        if row['team'] in self.DT_CUTOFF_DEFAULTS:
            last_day_of_year = self.DT_CUTOFF_DEFAULTS[row['team']]
        # If the cutoff does NOT exist for a team, but does exist for a league, replace last day of year with the league cutoff
        elif row['league'] in self.DT_CUTOFF_DEFAULTS:
            last_day_of_year = self.DT_CUTOFF_DEFAULTS[row['league']]
        else:
            last_day_of_year = self.DT_CUTOFF_DEFAULTS["last_day_of_year"]
       
    
        # If cancel date is this year, cutoff = cancel date + 14 (unless cancel date + 14 > last day of this year)
        if row["league"]=='NFL' and row["jersey_attribute"] == 'Game':
            in_days =  (last_day_of_year - self.CUR_DATE).days
            in_weeks = calculate_cutoff_in_weeks(in_days)
            return (in_days, in_weeks)
        
        elif (row['first_cancel_dt'] is not None) and (row['days_to_cancel_dt'] > 0 ) and ((row['days_to_cancel_dt'] + self.DEFAULT_DAYS_ADDITION) <= (last_day_of_year - self.CUR_DATE).days):
            in_days =  row['days_to_cancel_dt'] + self.DEFAULT_DAYS_ADDITION
            in_weeks = calculate_cutoff_in_weeks(in_days)
            return (in_days, in_weeks)

        # If cancel date was in past 30 days, cutoff = cancel date + 14 (unless 14 days out > last day of this year)
        elif (row['first_cancel_dt'] is not None) and (row['days_to_cancel_dt'] >= -self.MAX_CUTOFF_LOOKBACK_DAYS) and (self.DEFAULT_DAYS_ADDITION <= (last_day_of_year - self.CUR_DATE).days):
            in_days =  self.DEFAULT_DAYS_ADDITION
            in_weeks = calculate_cutoff_in_weeks(in_days)
            return (in_days, in_weeks)
        
        # If no active PO
        else:
            in_days =  (last_day_of_year - self.CUR_DATE).days
            in_weeks = calculate_cutoff_in_weeks(in_days)
            return (in_days, in_weeks)


    '''
    Calculate Blank Cutoff Feature 
    '''
    def get_blank_cut_off_v2(self, df_row):
        """
        Calculate Blank Cutoff Feature.
        
        Args:
        - df_row (Series): Row of DataFrame 
        
        Returns:
        - tuple: Tuple containing blank cutoff days and weeks.
        """
        # Helper function to calculate cutoff weeks from days
        def calculate_cutoff_in_weeks(days):
            if days % 7 == 0:
                return int(days / 7 + 1)
            else:
                return math.ceil(days / 7)


        if df_row['team'] in self.DT_CUTOFF_DEFAULTS:
            last_day_of_year  = self.DT_CUTOFF_DEFAULTS[row['team']]
            default_num_days  = (last_day_of_year - self.CUR_DATE).days
            default_in_weeks  = calculate_cutoff_in_weeks(num_days)

        # If the cutoff does NOT exist for a team, but does exist for a league, replace last day of year with the league cutoff
        elif df_row['league'] in self.DT_CUTOFF_DEFAULTS:
            last_day_of_year  = self.DT_CUTOFF_DEFAULTS[row['league']]
            default_num_days  = (last_day_of_year - self.CUR_DATE).days
            default_in_weeks  = calculate_cutoff_in_weeks(num_days)
        else: # Will use the last_day_of_year 
            default_num_days = (last_day_of_year - self.CUR_DATE).days #self.BLANK_DEFAULT_WEEKS_CUTOFF *7 
            default_in_weeks = self.BLANK_DEFAULT_WEEKS_CUTOFF
       

        if df_row["dm_sku"] is np.nan:
            return (np.nan, np.nan)
        
        elif (df_row["blank_first_cancel_date"] is not pd.NaT):
            return (df_row["blank_days_to_po"], calculate_cutoff_in_weeks(df_row["blank_days_to_po"]))
        
        else:
            return (default_num_days, default_in_weeks)

    # This logic is based on the including the cutoff dates at team level 
    def get_blank_cut_off(self, df_row):
        """
        Calculate Blank Cutoff Feature.
        
        Args:
        - df_row (Series): Row of DataFrame 
        
        Returns:
        - tuple: Tuple containing blank cutoff days and weeks.
        """
        # Helper function to calculate cutoff weeks from days
        def calculate_cutoff_in_weeks(days):
            if days % 7 == 0:
                return int(days / 7 + 1)
            else:
                return math.ceil(days / 7)


        if df_row['team'] in self.DT_CUTOFF_DEFAULTS:
            last_day_of_year = self.DT_CUTOFF_DEFAULTS[df_row['team']]
        # If the cutoff does NOT exist for a team, but does exist for a league, replace last day of year with the league cutoff
        elif df_row['league'] in self.DT_CUTOFF_DEFAULTS:
            last_day_of_year = self.DT_CUTOFF_DEFAULTS[df_row['league']]
        else:
            last_day_of_year = self.DT_CUTOFF_DEFAULTS["last_day_of_year"]
       
    


        # No DM_SKU Mapping available 
        if df_row["dm_sku"] is np.nan:
            return (np.nan, np.nan)
        
        # No DM_SKU Mapping available 
        elif df_row["league"] == 'NFL' and df_row["jersey_attribute"] == 'Game': 
            in_days =  (last_day_of_year - self.CUR_DATE).days
            in_weeks = calculate_cutoff_in_weeks(in_days)
            return (in_days, in_weeks)
        
        
        elif (df_row["blank_first_cancel_date"] is not pd.NaT):
            return (df_row["blank_days_to_po"], calculate_cutoff_in_weeks(df_row["blank_days_to_po"]))
        
        elif df_row["league"]=='NFL' and df_row['program_group_id'] == 'REPLEN-NO':
            num_days          = (self.NFL_END_OF_SEASON - self.CUR_DATE).days
            in_weeks          = calculate_cutoff_in_weeks(num_days)
            return (num_days, in_weeks)
        

        elif df_row["league"]=='MLB' and df_row['program_group_id'] == 'REPLEN-NO':
            num_days          = (self.MLB_END_OF_SEASON - self.CUR_DATE).days
            in_weeks          = calculate_cutoff_in_weeks(num_days)
            return (num_days, in_weeks)

        else:
            return (self.DEFAULT_WEEKS_CUTOFF * 7, self.DEFAULT_WEEKS_CUTOFF)



    def generate_blank_features(self, df_exc_mto):
        """
        Generate blank features based on the MTO data.
        
        Args:
        - df_exc_mto (DataFrame): DataFrame containing MTO data.
        
        Returns:
        - DataFrame: DataFrame with generated blank features.
        """
        def days_to_cancel(cancel_date, has_cancel_dt):
            if has_cancel_dt:
                if (cancel_date - self.CUR_DATE).days > 0:
                    return (cancel_date - self.CUR_DATE).days + self.DEFAULT_DAYS_ADDITION 
                else:  
                    return  self.DEFAULT_DAYS_ADDITION 
                # Parameter : 60 days  in past 
                #TODO : Need to have (cancel_date - self.CUR_DATE).days > -self.MAX_CUTOFF_LOOKBACK_DAYS_FOR_BLANKS:

            else:
                return None

        df_exc_mto["blank_first_cancel_date"] = pd.to_datetime(df_exc_mto["blank_first_cancel_date"])
        df_exc_mto["has_cancel_dt"] = df_exc_mto["blank_first_cancel_date"].apply(
            lambda x: False if x is pd.NaT else True)
        df_exc_mto["blank_days_to_po"] = df_exc_mto[["blank_first_cancel_date", "has_cancel_dt"]].apply(
            lambda tup: days_to_cancel(tup[0], tup[1]), axis=1)
        
        if self.BLANK_CUTOFF_METHOD == 'NEW':
            df_exc_mto['blank_cutoff_cols'] = df_exc_mto.apply(lambda df_row: self.get_blank_cut_off_v2(df_row), axis=1)
        elif self.BLANK_CUTOFF_METHOD=="OLD":
            df_exc_mto['blank_cutoff_cols'] = df_exc_mto.apply(lambda df_row: self.get_blank_cut_off(df_row), axis=1)




        df_exc_mto[["blank_cut_off", "blank_cut_off_weeks"]] = pd.DataFrame(df_exc_mto['blank_cutoff_cols'].tolist(),
                                                                             index=df_exc_mto.index)
        df_exc_mto = df_exc_mto.drop(columns=['blank_cutoff_cols'])

        return df_exc_mto 

    #TODO : Remove the function 
    def calculate_fallback_mto_units(self, df):
        """
        Calculate fallback MTO units.
        
        Args:
        - df (DataFrame): DataFrame containing relevant data.
        
        Returns:
        - DataFrame: DataFrame with calculated fallback MTO units.
        """
        df["fallback_units"] = df[["forecast_unadjusted", "ats_current"]].apply(
            lambda tup: tup[0] - tup[1] if tup[0] >= tup[1] else 0, axis=1)
        df["exc_fallback_units"] = df[["forecast_unadjusted", "ats_current"]].apply(
            lambda tup: tup[0] - tup[1] if tup[0] >= tup[1] else 0, axis=1)
        df_blank_fallback = df.groupby(["dm_sku"]).agg({"fallback_units": "sum"}).rename(
            columns={"fallback_units": "fallback_units_blank"}).reset_index(inplace=False)
        df = df.merge(df_blank_fallback, on="dm_sku", how="left")

        return df 

    def entry_code(self, df, df_mto_mapping):
        """
        Entry code to process data and generate features.
        
        Args:
        - df (DataFrame): Main data frame.
        - df_mto_mapping (DataFrame): DataFrame containing MTO mapping data.
        
        Returns:
        - DataFrame: Processed data frame with generated features.
        """
        df_subset = self.process_data_and_features(df, df_mto_mapping)
        df_subset = self.generate_blank_features(df_subset)
        df_subset['cutoff_in_weeks'] = df_subset[['raw_cutoff_in_weeks', "blank_cut_off_weeks"]].apply(
            lambda tup: int(min(tup)), axis=1)

        return df_subset
