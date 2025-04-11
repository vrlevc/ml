import unittest
from unittest.mock import Mock

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bank_account import BankAccount

class TestBankAccountIntegration(unittest.TestCase):
    def setUp(self):
        # Create a mock for the NotificationSystem
        self.mock_notification_system = Mock()
        # Pass the mock to the BankAccount instance
        self.account = BankAccount(100, notification_system=self.mock_notification_system)

    def test_deposit_calls_notification_system(self):
        # Perform a deposit
        self.account.deposit(50)

        # Verify that the notification system's notify method was called
        self.mock_notification_system.notify.assert_called_once_with(
            "Deposited 50, new balance: 150"
        )

    def test_invalid_deposit_does_not_call_notification_system(self):
        # Attempt to deposit an invalid amount
        with self.assertRaises(ValueError):
            self.account.deposit(-50)

        # Verify that the notification system's notify method was not called
        self.mock_notification_system.notify.assert_not_called()

    def test_withdraw_calls_notification_system(self):
        # Perform a withdrawal
        self.account.withdraw(50)

        # Verify that the notification system's notify method was called
        self.mock_notification_system.notify.assert_called_once_with(
            "Withdrew 50, new balance: 50"
        )

    def test_invalid_withdraw_does_not_call_notification_system(self):
        # Attempt to withdraw an invalid amount
        with self.assertRaises(ValueError):
            self.account.withdraw(-50)

        # Verify that the notification system's notify method was not called
        self.mock_notification_system.notify.assert_not_called()

        with self.assertRaises(ValueError):
            self.account.withdraw(200)  # Exceeds balance

        # Verify again that the notification system's notify method was not called
        self.mock_notification_system.notify.assert_not_called()

    def test_zero_balance_notification(self):
        # Withdraw the entire balance
        self.account.withdraw(100)

        # Verify that the notification system's notify method was called
        self.mock_notification_system.notify.assert_called_once_with(
            "Withdrew 100, new balance: 0"
        )

    def test_multiple_operations_notifications(self):
        # Perform multiple operations
        self.account.deposit(50)
        self.account.withdraw(30)

        # Verify that the notification system's notify method was called correctly
        self.mock_notification_system.notify.assert_any_call(
            "Deposited 50, new balance: 150"
        )
        self.mock_notification_system.notify.assert_any_call(
            "Withdrew 30, new balance: 120"
        )

    def test_no_notification_system(self):
        # Create a BankAccount without a notification system
        account = BankAccount(100)

        # Perform operations and ensure no exceptions are raised
        account.deposit(50)
        account.withdraw(30)

if __name__ == "__main__":
    unittest.main()