#!/bin/bash

# This script runs the BankAccount unit tests
python -m unittest test_bank_account.py

# This script runs the BankAccount integration tests
python -m unittest test_bank_account_integration.py