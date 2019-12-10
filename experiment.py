import os
import glob
import pprint
import redcap
import pandas as pd


# Open connection with REDCap server
redcap_token_path = os.path.join(os.path.expanduser(\"~\"), '.server_config/redcap-dataentry-token')
redcap_token_file = open(redcap_token_path, 'r')
redcap_token = redcap_token_file.read().strip()
redcap_project = redcap.Project('https://ncanda.sri.com/redcap/api/', redcap_token, verify_ssl=False)
datadict = redcap_project.export_metadata(format='df', df_kwargs=dict(index_col=None))

import gspread
from oauth2client.service_account import ServiceAccountCredentials

scope = ['https://spreadsheets.google.com/feeds']
credentials = ServiceAccountCredentials.from_json_keyfile_name('./wavi-eeg-screen-31ef6011b1d3.json', scope)
gc = gspread.authorize(credentials)

spreadsheet_key = '1cxtMAfEk4HjSPfa1aEvI2mcwSu2gI2a7U5OAwFFKxic'
book = gc.open_by_key(spreadsheet_key)
scheduling = book.worksheet(\"Scheduling\")
participants = book.worksheet(\"Participants\")
scheduling_table = scheduling.get_all_values()
participants_table = participants.get_all_values()
