import joblib
import uvicorn, os
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from lightgbm import LGBMClassifier
from additional_functions import *
from classes import *


app = FastAPI()

lgbm_loans = joblib.load('laon_lgbm.sav')
lgbm_grade = joblib.load('grade_lgbm.sav')
sub_grade_lgbm = joblib.load('sub_grade_lgbm.sav')


@app.get('/')
def home():
    return {'text': 'LendingClub loan predictions'}

@app.post("/loans_prediction")
async def create_application(loans_pred: Loan_prediction):

    loan_df = pd.DataFrame()

    if loans_pred.purpose not in purpose_dict:
        raise HTTPException(status_code=404, detail="Purpose not found")

    if loans_pred.addr_state not in addr_state_dict:
        raise HTTPException(status_code=404, detail="Address state not found")

    if loans_pred.emp_length not in emp_length_dict:
        raise HTTPException(status_code=404, detail="Employment length not found")

    loan_df['loan_amnt'] = [loans_pred.loan_amnt]
    loan_df['purpose'] = [loans_pred.purpose]
    loan_df['fico_range'] = [loans_pred.fico_range]
    loan_df['dti'] = [loans_pred.dti]
    loan_df['addr_state'] = [loans_pred.addr_state]
    loan_df['emp_length'] = [loans_pred.emp_length]
    loan_df['issue_year'] = [loans_pred.issue_year]
    loan_df['month'] = [loans_pred.month]

    loan_df = transformation_mnths(loan_df, 'month')
    loan_df = drop_columns(loan_df, 'month')
    prediction = lgbm_loans.predict(loan_df)
    if prediction[0] == 0:
        prediction = "Loan rejected"
    else:
        prediction = "Loan accepted"

    return {'prediction': prediction}

@app.post("/grade_prediction")
async def create_application(grad_pred: Grade_prediction):

    grade_df = pd.DataFrame()

    if grad_pred.purpose not in purpose_dict:
        raise HTTPException(status_code=404, detail="Purpose not found")

    if grad_pred.addr_state not in addr_state_dict:
        raise HTTPException(status_code=404, detail="Address state not found")

    if grad_pred.emp_length not in emp_length_dict:
        raise HTTPException(status_code=404, detail="Employment length not found")

    if grad_pred.emp_title not in emp_title_dict:
        raise HTTPException(status_code=404, detail="Employment title not found")

    if grad_pred.home_ownership not in home_ownership_dict:
        raise HTTPException(status_code=404, detail="Home ownership not found")

    if grad_pred.application_type not in application_type_dict:
        raise HTTPException(status_code=404, detail="Application type not found")

    if grad_pred.hardship_status not in hardship_status_dict:
        raise HTTPException(status_code=404, detail="Hardship status not found")

    grade_df['loan_amnt'] = [grad_pred.loan_amnt]
    grade_df['funded_amnt'] = [grad_pred.funded_amnt]
    grade_df['funded_amnt_inv'] = [grad_pred.funded_amnt_inv]
    grade_df['term'] = [grad_pred.term]
    grade_df['annual_inc'] = [grad_pred.annual_inc]
    grade_df['dti'] = [grad_pred.dti]
    grade_df['delinq_2yrs'] = [grad_pred.delinq_2yrs]
    grade_df['open_acc'] = [grad_pred.open_acc]
    grade_df['pub_rec'] = [grad_pred.pub_rec]
    grade_df['revol_bal'] = [grad_pred.revol_bal]
    grade_df['revol_util'] = [grad_pred.revol_util]
    grade_df['total_acc'] = [grad_pred.total_acc]
    grade_df['annual_inc_joint'] = [grad_pred.annual_inc_joint]
    grade_df['dti_joint'] = [grad_pred.dti_joint]
    grade_df['tot_coll_amt'] = [grad_pred.tot_coll_amt]
    grade_df['tot_cur_bal'] = [grad_pred.tot_cur_bal]
    grade_df['open_acc_6m'] = [grad_pred.open_acc_6m]
    grade_df['open_rv_24m'] = [grad_pred.open_rv_24m]
    grade_df['max_bal_bc'] = [grad_pred.max_bal_bc]
    grade_df['all_util'] = [grad_pred.all_util]
    grade_df['total_cu_tl'] = [grad_pred.total_cu_tl]
    grade_df['inq_last_12m'] = [grad_pred.inq_last_12m]
    grade_df['avg_cur_bal'] = [grad_pred.avg_cur_bal]
    grade_df['bc_open_to_buy'] = [grad_pred.bc_open_to_buy]
    grade_df['bc_util'] = [grad_pred.bc_util]
    grade_df['mort_acc'] = [grad_pred.mort_acc]
    grade_df['num_bc_sats'] = [grad_pred.num_bc_sats]
    grade_df['num_bc_tl'] = [grad_pred.num_bc_tl]
    grade_df['num_rev_accts'] = [grad_pred.num_rev_accts]
    grade_df['num_sats'] = [grad_pred.num_sats]
    grade_df['percent_bc_gt_75'] = [grad_pred.percent_bc_gt_75]
    grade_df['pub_rec_bankruptcies'] = [grad_pred.pub_rec_bankruptcies]
    grade_df['hardship_amount'] = [grad_pred.hardship_amount]
    grade_df['issue_year'] = [grad_pred.issue_year]
    grade_df['issue_month'] = [grad_pred.issue_month]
    grade_df['fico_range'] = [grad_pred.fico_range]
    grade_df['purpose'] = [grad_pred.purpose]
    grade_df['emp_length'] = [grad_pred.emp_length]
    grade_df['emp_title'] = [grad_pred.emp_title]
    grade_df['home_ownership'] = [grad_pred.home_ownership]
    grade_df['application_type'] = [grad_pred.application_type]
    grade_df['hardship_status'] = [grad_pred.hardship_status]
    grade_df['addr_state'] = [grad_pred.addr_state]

    grade_df = transformation_mnths(grade_df, 'issue_month')
    grade_df = drop_columns(grade_df, 'issue_month')
    prediction = lgbm_grade.predict(grade_df)
    answer = (f'You will get a grade: {prediction}')

    return {'prediction': answer}

@app.post("/sub_grade_prediction")
async def create_application(grad_pred: Grade_prediction):

    sub_grade_df = pd.DataFrame()

    if grad_pred.purpose not in purpose_dict:
        raise HTTPException(status_code=404, detail="Purpose not found")

    if grad_pred.addr_state not in addr_state_dict:
        raise HTTPException(status_code=404, detail="Address state not found")

    if grad_pred.emp_length not in emp_length_dict:
        raise HTTPException(status_code=404, detail="Employment length not found")

    if grad_pred.emp_title not in emp_title_dict:
        raise HTTPException(status_code=404, detail="Employment title not found")

    if grad_pred.home_ownership not in home_ownership_dict:
        raise HTTPException(status_code=404, detail="Home ownership not found")

    if grad_pred.application_type not in application_type_dict:
        raise HTTPException(status_code=404, detail="Application type not found")

    if grad_pred.hardship_status not in hardship_status_dict:
        raise HTTPException(status_code=404, detail="Hardship status not found")

    sub_grade_df['loan_amnt'] = [grad_pred.loan_amnt]
    sub_grade_df['funded_amnt'] = [grad_pred.funded_amnt]
    sub_grade_df['funded_amnt_inv'] = [grad_pred.funded_amnt_inv]
    sub_grade_df['term'] = [grad_pred.term]
    sub_grade_df['annual_inc'] = [grad_pred.annual_inc]
    sub_grade_df['dti'] = [grad_pred.dti]
    sub_grade_df['delinq_2yrs'] = [grad_pred.delinq_2yrs]
    sub_grade_df['open_acc'] = [grad_pred.open_acc]
    sub_grade_df['pub_rec'] = [grad_pred.pub_rec]
    sub_grade_df['revol_bal'] = [grad_pred.revol_bal]
    sub_grade_df['revol_util'] = [grad_pred.revol_util]
    sub_grade_df['total_acc'] = [grad_pred.total_acc]
    sub_grade_df['annual_inc_joint'] = [grad_pred.annual_inc_joint]
    sub_grade_df['dti_joint'] = [grad_pred.dti_joint]
    sub_grade_df['tot_coll_amt'] = [grad_pred.tot_coll_amt]
    sub_grade_df['tot_cur_bal'] = [grad_pred.tot_cur_bal]
    sub_grade_df['open_acc_6m'] = [grad_pred.open_acc_6m]
    sub_grade_df['open_rv_24m'] = [grad_pred.open_rv_24m]
    sub_grade_df['max_bal_bc'] = [grad_pred.max_bal_bc]
    sub_grade_df['all_util'] = [grad_pred.all_util]
    sub_grade_df['total_cu_tl'] = [grad_pred.total_cu_tl]
    sub_grade_df['inq_last_12m'] = [grad_pred.inq_last_12m]
    sub_grade_df['avg_cur_bal'] = [grad_pred.avg_cur_bal]
    sub_grade_df['bc_open_to_buy'] = [grad_pred.bc_open_to_buy]
    sub_grade_df['bc_util'] = [grad_pred.bc_util]
    sub_grade_df['mort_acc'] = [grad_pred.mort_acc]
    sub_grade_df['num_bc_sats'] = [grad_pred.num_bc_sats]
    sub_grade_df['num_bc_tl'] = [grad_pred.num_bc_tl]
    sub_grade_df['num_rev_accts'] = [grad_pred.num_rev_accts]
    sub_grade_df['num_sats'] = [grad_pred.num_sats]
    sub_grade_df['percent_bc_gt_75'] = [grad_pred.percent_bc_gt_75]
    sub_grade_df['pub_rec_bankruptcies'] = [grad_pred.pub_rec_bankruptcies]
    sub_grade_df['hardship_amount'] = [grad_pred.hardship_amount]
    sub_grade_df['issue_year'] = [grad_pred.issue_year]
    sub_grade_df['issue_month'] = [grad_pred.issue_month]
    sub_grade_df['fico_range'] = [grad_pred.fico_range]
    sub_grade_df['purpose'] = [grad_pred.purpose]
    sub_grade_df['emp_length'] = [grad_pred.emp_length]
    sub_grade_df['emp_title'] = [grad_pred.emp_title]
    sub_grade_df['home_ownership'] = [grad_pred.home_ownership]
    sub_grade_df['application_type'] = [grad_pred.application_type]
    sub_grade_df['hardship_status'] = [grad_pred.hardship_status]
    sub_grade_df['addr_state'] = [grad_pred.addr_state]

    sub_grade_df = transformation_mnths(sub_grade_df, 'issue_month')
    sub_grade_df = drop_columns(sub_grade_df, 'issue_month')
    prediction = sub_grade_lgbm.predict(sub_grade_df)
    answer = (f'You will get a sub-grade: {prediction}')

    return {'prediction': answer}




