import pandas as pd
from math import pi
import math
from pydantic import BaseModel


def transformation_mnths(df:pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Takes list of month column values.
    Converts them to sin and coss values.
    Returns two new lists.
    """
    max_value = df[column].max()
    df['sin_mnths'] = [math.sin((2*pi*x)/max_value) for x in list(df[column])]
    df['cos_mnths'] = [math.cos((2*pi*x)/max_value) for x in list(df[column])]

    return df

def drop_columns (df: pd.DataFrame, columns: list) -> pd.DataFrame:

    df = df.drop(columns=columns)

    return df

class Loan_prediction(BaseModel):
    loan_amnt: float
    purpose: str
    fico_range: float
    dti: float
    addr_state: str
    emp_length:str
    issue_year: int
    month: int

class Grade_prediction(BaseModel):
    loan_amnt: float
    funded_amnt: float
    funded_amnt_inv: float
    term: float
    annual_inc: float
    dti: float
    delinq_2yrs: float
    open_acc: float
    open_acc: float
    pub_rec: float
    revol_bal: float
    revol_util: float
    total_acc: float
    annual_inc_joint: float
    dti_joint: float
    tot_coll_amt: float
    tot_cur_bal: float
    open_acc_6m: float
    open_rv_24m: float
    max_bal_bc: float
    all_util: float
    total_cu_tl: float
    inq_last_12m: float
    avg_cur_bal: float
    bc_open_to_buy: float
    bc_util: float
    mort_acc: float
    num_bc_sats: float
    num_bc_tl: float
    num_rev_accts: float
    num_sats: float
    percent_bc_gt_75: float
    pub_rec_bankruptcies: float
    hardship_amount: float
    issue_year: float
    issue_month: float
    fico_range: float
    purpose: str
    emp_length: str
    emp_title: str
    home_ownership: str
    application_type: str
    hardship_status: str
    addr_state: str

purpose_dict = {
  "other": "other",
  "debt_consolidation": "debt_consolidation",
  "vacation": "vacation",
  "car": "car",
  "major_purchase": "major_purchase",
  "credit_card": "credit_card",
  "home_improvement": "home_improvement",
  "small_business": "small_business",
  "medical": "medical",
  "moving": "moving",
  "renewable_energy": "renewable_energy",
  "educational": "educational",
  "wedding": "wedding",
}

addr_state_dict = {
  "NJ": "NJ",
  "NY": "NY",
  "TX": "TX",
  "WA": "WA",
  "MO": "MO",
  "AL": "AL",
  "GA": "GA",
  "VA": "VA",
  "OH": "OH",
  "LA": "LA",
  "IN": "IN",
  "FL": "FL",
  "SC": "SC",
  "MN": "MN",
  "NV": "NV",
  "TN": "TN",
  "MD": "MD",
  "MI": "MI",
  "PA": "PA",
  "MA": "MA",
  "NE": "NE",
  "ID": "ID",
  "NH": "NH",
  "KY": "KY",
  "NM": "NM",
  "UT": "UT",
  "MT": "MT",
  "RI": "RI",
  "VT": "VT",
  "DC": "DC",
  "AR": "AR",
  "HI": "HI",
  "AK": "AK",
  "WY": "WY",
  "ND": "ND",
  "SD": "SD",
  "IA": "IA",
}

emp_length_dict = {
  "< 1 year": "< 1 year",
  "1 year": "1 year",
  "2 years": "2 years",
  "3 years": "3 years",
  "4 years": "4 years",
  "5 years": "5 years",
  "6 years": "6 years",
  "7 years": "7 years",
  "8 years": "8 years",
  "9 years": "9 years",
  "10+ years": "10+ years",
}

emp_title_dict = {
  "Engineering": "Engineering",
  "Government": "Government",
  "Maintenance, workers and transport": "Maintenance, workers and transport",
  "Else": "Else",
  "Science and technology": "Science and technology",
  "Arts, culture and entertainment": "Arts, culture and entertainment",
  "Education": "Education",
  "Law": "Law",
  "Health and medicine": "Health and medicine"
}

home_ownership_dict = {
  "MORTGAGE": "MORTGAGE",
  "RENT": "RENT",
  "OWN": "OWN",
  "ANY": "ANY"
}

application_type_dict = {
  "Individual": "Individual",
  "Joint App": "Joint App"
}

hardship_status_dict = {
  "uknown": "uknown",
  "COMPLETED": "COMPLETED",
  "ACTIVE": "ACTIVE",
  "BROKEN": "BROKEN"
}