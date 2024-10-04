import unittest

from gradescope_utils.autograder_utils.decorators import weight, number, partial_credit

import os
import pandas as pd
import numpy as np
import requests

from utils import (
        extract_variables, 
        extract_initial_variables, 
        find_cells_with_text, 
        find_cells_by_indices,
        has_string_in_cell,
        has_string_in_code_cells,
        extract_cell_content_and_outputs,
        search_in_extracted_content,
        print_text_and_output_cells,
        print_code_and_output_cells)


class GradeAssignment(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(GradeAssignment, self).__init__(*args, **kwargs)
        self.notebook_path = None


    @weight(0.0)
    @number("2.1")
    def test_cute_webscraping(self):
        print('')

        begin_cells = find_cells_with_text(self.notebook_path, "TASK 1.0: Cute Webscraping (5 points)")
        begin_cell = begin_cells[0]
        begin_cell_idx = begin_cell['index'] - 1

        end_cells = find_cells_with_text(self.notebook_path, "TASK 1.1: 1 Liner Thingz (3 points)")
        end_cell = end_cells[0]
        end_cell_idx = end_cell['index']
        cell_vars = extract_variables(self.notebook_path, cell_idx=end_cell_idx - 1)


        df_from_code = cell_vars.get("df", None)
        if os.path.exists("volcanoes.csv"):
            df_from_csv = pd.read_csv("volcanoes.csv")
            df_from_csv = df_from_csv[['id', 'year', 'month', 'day', 'tsunamiEventId', 'earthquakeEventId', 'volcanoLocationId', 'volcanoLocationNewNum', 'volcanoLocationNum', 'name', 'location', 'country', 'latitude', 'longitude', 'elevation', 'morphology', 'agent', 'deathsTotal', 'deathsAmountOrderTotal', 'damageAmountOrderTotal', 'significant', 'publish', 'eruption', 'status', 'timeErupt', 'vei', 'deathsAmountOrder', 'damageAmountOrder', 'housesDestroyedAmountOrderTotal', 'deaths', 'injuries', 'injuriesAmountOrder', 'injuriesTotal', 'injuriesAmountOrderTotal', 'housesDestroyedAmountOrder', 'housesDestroyed', 'housesDestroyedTotal', 'missingAmountOrder', 'missingAmountOrderTotal', 'missing', 'missingTotal', 'damageMillionsDollars', 'damageMillionsDollarsTotal']]
        else:
            df_from_csv = None

        if True:
            api_url = "https://www.ngdc.noaa.gov/hazel/hazard-service/api/v1/volcanoes"
            headers = {
                'accept': '*/*'
            }
            response = requests.get(api_url, headers=headers)
            data = response.json()
            items = data['items']

            df_gt = pd.DataFrame(items)


        exists_df = (df_from_code is not None) and (type(df_from_code) == pd.DataFrame)
        exists_csv = (df_from_csv is not None) and (type(df_from_csv) == pd.DataFrame)

        equal_df_csv = df_from_code.equals(df_from_csv)
        equal_df_gt = df_from_code.equals(df_gt)

        print("Exists df in the code: ", exists_df)
        print('Exists "volcanoes.csv": ', exists_csv)
        print("Equal df and csv: ", equal_df_csv)
        print("Equal df and gt: ", equal_df_gt)

        result = exists_df and exists_csv and equal_df_csv and equal_df_gt
        self.assertTrue(result)

    @partial_credit(1.0)
    @number("2.2")
    def test_linear_thingz_1_1_1(self, set_score=None):
        print('')


        begin_cells = find_cells_with_text(self.notebook_path, "**1.1.1:** *In one line of code and **using only one function**")
        begin_cell = begin_cells[0]
        begin_cell_idx = begin_cell['index']

        end_cells = find_cells_with_text(self.notebook_path, "**1.1.2:** *In one line of code, list the **names** of all the **features** in the dataframe.*")
        end_cell = end_cells[0]
        end_cell_idx = end_cell['index']

        cell_vars = extract_variables(self.notebook_path, cell_idx=end_cell_idx - 1)
        cell_texts = extract_cell_content_and_outputs(self.notebook_path, begin_cell_idx, end_cell_idx)

        # search for shape, 200, 43
        search_shape, _ = search_in_extracted_content(cell_texts, "shape")
        search_200, _ = search_in_extracted_content(cell_texts, "200")
        search_43, _ = search_in_extracted_content(cell_texts, "43")

        found_shape_fn = search_shape
        found_shape_val = search_200 and search_43

        print("Found shape function: ", found_shape_fn)
        print("Found shape values: ", found_shape_val)

        if found_shape_val:
            set_score(0.5)
            if found_shape_fn:
                set_score(1.0)
        else:
            set_score(0.0)

    @partial_credit(1.0)
    @number("2.3")
    def test_linear_thingz_1_1_2(self, set_score=None):
        print('')


        begin_cells = find_cells_with_text(self.notebook_path, "**1.1.2:** *In one line of code, list the **names** of all the **features** in the dataframe.*")
        begin_cell = begin_cells[0]
        begin_cell_idx = begin_cell['index']

        end_cells = find_cells_with_text(self.notebook_path, "**1.1.3:** *In one line of code, create a **new dataframe** called **new_df** that **contains all** the features of the **old** dataframe **except the following**:*")
        end_cell = end_cells[0]
        end_cell_idx = end_cell['index']

        cell_texts = extract_cell_content_and_outputs(self.notebook_path, begin_cell_idx, end_cell_idx)

        search_columns, _ = search_in_extracted_content(cell_texts, "columns")

        search_feature_year, _ = search_in_extracted_content(cell_texts, "year")
        search_feature_housesDestroyedAmountOrderTotal, _ = search_in_extracted_content(cell_texts, "housesDestroyedAmountOrderTotal")
        search_feature_damageMillionsDollarsTotal, _ = search_in_extracted_content(cell_texts, "damageMillionsDollarsTotal")
        search_feature = search_feature_year and search_feature_housesDestroyedAmountOrderTotal and search_feature_damageMillionsDollarsTotal


        found_columns_fn = search_columns
        found_features_name = search_feature

        print("Found columns function: ", found_columns_fn)
        print("Found features names: ", found_features_name)


        if found_features_name:
            set_score(0.5)
            if found_columns_fn:
                set_score(1.0)
        else:
            set_score(0.0)


    @partial_credit(1.0)
    @number("2.4")
    def test_linear_thingz_1_1_3(self, set_score=None):
        print('')

        begin_cells = find_cells_with_text(self.notebook_path, "**1.1.3:** *In one line of code, create a **new dataframe** called **new_df** that **contains all** the features of the **old** dataframe **except the following**:*")
        begin_cell = begin_cells[0]
        begin_cell_idx = begin_cell['index']

        end_cells = find_cells_with_text(self.notebook_path, "#### **TASK 1.2: 1 Liner Shenaniganz (7 points)**")
        end_cell = end_cells[0]
        end_cell_idx = end_cell['index']

        cell_vars = extract_variables(self.notebook_path, cell_idx=end_cell_idx - 1)
        cell_texts = extract_cell_content_and_outputs(self.notebook_path, begin_cell_idx, end_cell_idx)
        search_drop, _ = search_in_extracted_content(cell_texts, "drop")

        new_df = cell_vars.get("new_df", None)

        excluded_columns = [
            'volcanoLocationNum', 'location', 'latitude', 'longitude', 'agent', 
            'significant', 'publish', 'eruption', 'status', 'timeErupt', 'damageAmountOrder', 
            'damageAmountOrderTotal', 'housesDestroyedAmountOrder', 'housesDestroyedAmountOrderTotal', 
            'housesDestroyed', 'housesDestroyedTotal', 'missingAmountOrder', 'missingAmountOrderTotal', 
            'missing', 'missingTotal', 'damageMillionsDollars', 'damageMillionsDollarsTotal', 
            'injuries', 'injuriesAmountOrder', 'injuriesTotal', 'injuriesAmountOrderTotal', 
            'deathsAmountOrderTotal', 'deathsAmountOrder'
        ]

        # Calculate flag: True if all required columns are present in new_df, False otherwise

        exists_new_df = (new_df is not None) and (type(new_df) == pd.DataFrame)

        if exists_new_df:
            excluded_columns = not any(col in new_df.columns for col in excluded_columns)
        else:
            excluded_columns = False


        print("Exists new_df: ", exists_new_df)
        print("Excluded columns correctly: ", excluded_columns)
        print("Used drop function: ", search_drop)

        set_score(0.0)
        if exists_new_df:
            if excluded_columns:
                set_score(0.5)
                if search_drop:
                    set_score(1.0)
        




    @partial_credit(2.0)
    @number("2.5")
    def test_Liner_Shenaniganz_1_2_1(self, set_score=None):
        print('')

        begin_cells = find_cells_with_text(self.notebook_path, "#### **TASK 1.2: 1 Liner Shenaniganz (7 points)**")
        begin_cell = begin_cells[0]
        begin_cell_idx = begin_cell['index']

        end_cells = find_cells_with_text(self.notebook_path, "**1.2.2:** *In one line of code, **reset** the **index column** of the dataframe so that it has **1-based indexing**.*")
        end_cell = end_cells[0]
        end_cell_idx = end_cell['index']

        cell_vars = extract_variables(self.notebook_path, cell_idx=end_cell_idx - 1)
        cell_texts = extract_cell_content_and_outputs(self.notebook_path, begin_cell_idx, end_cell_idx)

        search_dropna, _ = search_in_extracted_content(cell_texts, "dropna")
        new_df = cell_vars.get("new_df", None)

        exists_new_df = (new_df is not None) and (type(new_df) == pd.DataFrame)
            
        contains_nan = True
        if exists_new_df:
            # Check whether new_df contains NaN in any of the "year", "month", "day" columns
            contains_nan = new_df[['year', 'month', 'day']].isna().any().any()

        print("Exists new_df: ", exists_new_df)
        print("Contains NaN: ", contains_nan)
        print("Used dropna function: ", search_dropna)  

        set_score(0.0)
        if exists_new_df:
            if search_dropna:
                set_score(1.0)

            if not contains_nan:
                set_score(2.0)

    @partial_credit(2.0)
    @number("2.6")
    def test_Liner_Shenaniganz_1_2_2(self, set_score=None):
        print('')

        begin_cells = find_cells_with_text(self.notebook_path, "**1.2.2:** *In one line of code, **reset** the **index column** of the dataframe so that it has **1-based indexing**.*")
        begin_cell = begin_cells[0]
        begin_cell_idx = begin_cell['index']

        end_cells = find_cells_with_text(self.notebook_path, "**1.2.3:** *In one line of code, make a **new column** called **'totalDeaths'** that takes the **max** of the values given between")
        end_cell = end_cells[0]
        end_cell_idx = end_cell['index']

        cell_vars_prev = extract_variables(self.notebook_path, cell_idx=begin_cell_idx)
        cell_vars = extract_variables(self.notebook_path, cell_idx=end_cell_idx - 1)

        new_df = cell_vars.get("new_df", None)
        new_df_prev = cell_vars_prev.get("new_df", None)

        exists_new_df = (new_df is not None) and (type(new_df) == pd.DataFrame)

        min_idx_prev = new_df_prev.index.min()
        max_idx_prev = new_df_prev.index.max()

        min_idx = new_df.index.min()
        max_idx = new_df.index.max()
        len_idx = len(new_df.index)


        min_idx_changed = min_idx != min_idx_prev
        max_idx_changed = max_idx != max_idx_prev

        idx_changed = min_idx_changed or max_idx_changed
        idx_reindexd = (min_idx == 1) and (len_idx == max_idx)
 
        print("Exists new_df: ", exists_new_df)
        print("Index changed: ", idx_changed)
        print("Indexing is 1-based: ", idx_reindexd)

        set_score(0.0)
        if exists_new_df:
            if idx_changed:
                set_score(1.0)
            if idx_reindexd:
                set_score(2.0)

    @partial_credit(3.0)
    @number("2.7")
    def test_Liner_Shenaniganz_1_2_3(self, set_score=None):
        print('')

        begin_cells = find_cells_with_text(self.notebook_path, "**1.2.3:** *In one line of code, make a **new column** called **'totalDeaths'** that takes the **max** of the values given between")
        begin_cell = begin_cells[0]
        begin_cell_idx = begin_cell['index']

        end_cells = find_cells_with_text(self.notebook_path, "#### **TASK 1.3: Tailoring Time (10 Points)**")
        end_cell = end_cells[0]
        end_cell_idx = end_cell['index']

        cell_vars = extract_variables(self.notebook_path, cell_idx=end_cell_idx - 1)

        new_df = cell_vars.get("new_df", None)

        exists_new_df = (new_df is not None) and (type(new_df) == pd.DataFrame)

        has_totalDeaths = "totalDeaths" in new_df.keys()

        totalDeaths_gt = new_df[['deathsTotal', 'deaths']].max(axis=1, skipna=True)
        totalDeaths = new_df["totalDeaths"]

        correct_totalDeaths = (totalDeaths_gt.equals(totalDeaths))

        print("Exists new_df: ", exists_new_df)
        print("Has totalDeaths: ", has_totalDeaths)
        print("Correct totalDeaths: ", correct_totalDeaths)

        set_score(0.0)
        if exists_new_df:
            if has_totalDeaths:
                set_score(1.0)
                if correct_totalDeaths:
                    set_score(3.0)

    @partial_credit(10.0)
    @number("2.8")
    def test_Tailoring_Time_1_3(self, set_score=None):
        print('')

        begin_cells = find_cells_with_text(self.notebook_path, "#### **TASK 1.3: Tailoring Time (10 Points)**")
        begin_cell = begin_cells[0]
        begin_cell_idx = begin_cell['index']

        end_cells = find_cells_with_text(self.notebook_path, "**Part 2: Volcanic Matryoshkas")
        end_cell = end_cells[0]
        end_cell_idx = end_cell['index']

        cell_vars = extract_variables(self.notebook_path, cell_idx=end_cell_idx - 1)

        new_df = cell_vars.get("new_df", None)

        exists_new_df = (new_df is not None) and (type(new_df) == pd.DataFrame)

        date_column_exists = "date" in new_df.keys()

        def check_date_format(df, column_name):
            correct_format = []
            for date in df[column_name]:
                try:
                    # Check if it can be converted to datetime
                    pd.to_datetime(date, format='%Y-%m-%d', errors='raise')
                    correct_format.append(True)
                except (ValueError, TypeError):
                    correct_format.append(False)
            return correct_format

        date_column_correct_format = all(check_date_format(new_df, "date"))

        year_column_exists = "year" in new_df.keys()
        month_column_exists = "month" in new_df.keys()
        day_column_exists = "day" in new_df.keys()

        date_column_next_to_id = new_df.keys()[1] == "date"

        print("Exists new_df: ", exists_new_df)
        print("Date column exists: ", date_column_exists)
        print("Date column has correct format: ", date_column_correct_format)
        print("Date column next to id: ", date_column_next_to_id)
        print("Year column exists: ", year_column_exists)
        print("Month column exists: ", month_column_exists)
        print("Day column exists: ", day_column_exists)

        total_score = 0.0

        if exists_new_df:
            if date_column_exists:
                total_score = 10.0

                if month_column_exists or year_column_exists or day_column_exists:
                    total_score -= 2.0

                if not date_column_correct_format:
                    total_score -= 3.0

                if not date_column_next_to_id:
                    total_score -= 3.0

        set_score(total_score)
