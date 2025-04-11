import unittest

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bank_account import BankAccount

class TestBankAccount(unittest.TestCase):
    def setUp(self):
        # Common setup for each test
        self.account = BankAccount(100)

    def test_initial_balance_positive(self):
        self.assertEqual(self.account.get_balance(), 100)

    def test_initial_balance_zero(self):
        account = BankAccount()
        self.assertEqual(account.get_balance(), 0)

    def test_initial_balance_negative(self):
        with self.assertRaises(ValueError):
            BankAccount(-50)

    def test_deposit_positive_amount(self):
        self.account.deposit(50)
        self.assertEqual(self.account.get_balance(), 150)

    def test_deposit_zero_amount(self):
        with self.assertRaises(ValueError):
            self.account.deposit(0)

    def test_deposit_negative_amount(self):
        with self.assertRaises(ValueError):
            self.account.deposit(-50)

    def test_withdraw_valid_amount(self):
        self.account.withdraw(50)
        self.assertEqual(self.account.get_balance(), 50)

    def test_withdraw_zero_amount(self):
        with self.assertRaises(ValueError):
            self.account.withdraw(0)

    def test_withdraw_negative_amount(self):
        with self.assertRaises(ValueError):
            self.account.withdraw(-50)

    def test_withdraw_more_than_balance(self):
        with self.assertRaises(ValueError):
            self.account.withdraw(150)

    def test_multiple_operations(self):
        self.account.deposit(50)
        self.account.withdraw(30)
        self.assertEqual(self.account.get_balance(), 120)

if __name__ == "__main__":
    unittest.main()