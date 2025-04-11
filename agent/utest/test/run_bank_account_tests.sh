#!/bin/bash

# Ensure the script runs tests in the current directory
python -m unittest test_bank_account.py
python -m unittest test_bank_account_integration.py