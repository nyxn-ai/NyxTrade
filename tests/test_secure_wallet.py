"""
Unit Tests for Secure Wallet Manager
Tests wallet security, encryption, and private key protection
"""

import pytest
import asyncio
import tempfile
import os
from unittest.mock import Mock, patch, AsyncMock

from utils.secure_wallet import (
    SecureWalletManager, WalletType, WalletInfo, 
    WalletSecurityValidator, PrivateKeyExposureException
)
from utils.exceptions import WalletSecurityException, EncryptionException


class TestSecureWalletManager:
    """Test cases for SecureWalletManager"""
    
    @pytest.fixture
    def wallet_manager(self):
        """Create wallet manager for testing"""
        return SecureWalletManager("test_password_123")
    
    @pytest.fixture
    def mock_keyring(self):
        """Mock keyring for testing"""
        with patch('utils.secure_wallet.keyring') as mock:
            mock.get_password.return_value = None
            mock.set_password.return_value = None
            mock.delete_password.return_value = None
            yield mock
    
    def test_initialization_with_password(self):
        """Test wallet manager initialization with password"""
        manager = SecureWalletManager("test_password")
        assert manager.encryption_key is not None
        assert manager.master_password == "test_password"
    
    def test_initialization_without_password(self):
        """Test wallet manager initialization without password"""
        manager = SecureWalletManager()
        assert manager.encryption_key is None
        assert manager.master_password is None
    
    def test_create_local_wallet(self, wallet_manager, mock_keyring):
        """Test creating local encrypted wallet"""
        wallet_info = wallet_manager.create_wallet(
            "test_wallet",
            WalletType.LOCAL_ENCRYPTED,
            "ethereum"
        )
        
        assert wallet_info.wallet_id == "test_wallet"
        assert wallet_info.wallet_type == WalletType.LOCAL_ENCRYPTED
        assert wallet_info.network == "ethereum"
        assert wallet_info.address.startswith("0x")
        assert len(wallet_info.address) == 42
        
        # Verify private key was encrypted and stored
        mock_keyring.set_password.assert_called()
    
    def test_create_hd_wallet(self, wallet_manager, mock_keyring):
        """Test creating HD wallet"""
        wallet_info = wallet_manager.create_wallet(
            "test_hd_wallet",
            WalletType.MNEMONIC_HD,
            "ethereum"
        )
        
        assert wallet_info.wallet_id == "test_hd_wallet"
        assert wallet_info.wallet_type == WalletType.MNEMONIC_HD
        assert wallet_info.address.startswith("0x")
        
        # Verify both mnemonic and private key were stored
        assert mock_keyring.set_password.call_count >= 2
    
    def test_create_hardware_wallet(self, wallet_manager):
        """Test hardware wallet setup"""
        wallet_info = wallet_manager.create_wallet(
            "test_hardware",
            WalletType.HARDWARE_LEDGER,
            "ethereum"
        )
        
        assert wallet_info.wallet_id == "test_hardware"
        assert wallet_info.wallet_type == WalletType.HARDWARE_LEDGER
    
    def test_wallet_creation_without_encryption(self):
        """Test wallet creation fails without encryption"""
        manager = SecureWalletManager()  # No password
        
        with pytest.raises(ValueError, match="Encryption not initialized"):
            manager.create_wallet("test", WalletType.LOCAL_ENCRYPTED, "ethereum")
    
    def test_get_wallet_info(self, wallet_manager, mock_keyring):
        """Test getting wallet information"""
        # Create wallet first
        wallet_info = wallet_manager.create_wallet(
            "test_wallet",
            WalletType.LOCAL_ENCRYPTED,
            "ethereum"
        )
        
        # Get wallet info
        retrieved_info = wallet_manager.get_wallet_info("test_wallet")
        
        assert retrieved_info is not None
        assert retrieved_info.wallet_id == wallet_info.wallet_id
        assert retrieved_info.address == wallet_info.address
    
    def test_list_wallets(self, wallet_manager, mock_keyring):
        """Test listing wallets"""
        # Create multiple wallets
        wallet_manager.create_wallet("wallet1", WalletType.LOCAL_ENCRYPTED, "ethereum")
        wallet_manager.create_wallet("wallet2", WalletType.MNEMONIC_HD, "ethereum")
        
        wallets = wallet_manager.list_wallets()
        
        assert len(wallets) == 2
        wallet_ids = [w.wallet_id for w in wallets]
        assert "wallet1" in wallet_ids
        assert "wallet2" in wallet_ids
    
    def test_get_public_address(self, wallet_manager, mock_keyring):
        """Test getting public address"""
        wallet_info = wallet_manager.create_wallet(
            "test_wallet",
            WalletType.LOCAL_ENCRYPTED,
            "ethereum"
        )
        
        address = wallet_manager.get_public_address("test_wallet")
        assert address == wallet_info.address
        assert address.startswith("0x")
    
    def test_get_nonexistent_wallet(self, wallet_manager):
        """Test getting non-existent wallet"""
        wallet_info = wallet_manager.get_wallet_info("nonexistent")
        assert wallet_info is None
    
    def test_delete_wallet(self, wallet_manager, mock_keyring):
        """Test deleting wallet"""
        # Create wallet
        wallet_manager.create_wallet("test_wallet", WalletType.LOCAL_ENCRYPTED, "ethereum")
        
        # Delete wallet
        wallet_manager.delete_wallet("test_wallet")
        
        # Verify wallet is deleted
        wallet_info = wallet_manager.get_wallet_info("test_wallet")
        assert wallet_info is None
        
        # Verify keyring deletion was called
        mock_keyring.delete_password.assert_called()


class TestWalletSecurityValidator:
    """Test cases for WalletSecurityValidator"""
    
    def test_validate_safe_data(self):
        """Test validation of safe data"""
        safe_data = {
            "wallet_id": "test_wallet",
            "address": "0x1234567890123456789012345678901234567890",
            "balance": 1.5
        }
        
        assert WalletSecurityValidator.validate_no_private_key_exposure(safe_data) is True
    
    def test_detect_private_key_string(self):
        """Test detection of private key in string"""
        private_key = "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef"
        
        assert WalletSecurityValidator.validate_no_private_key_exposure(private_key) is False
    
    def test_detect_private_key_in_dict(self):
        """Test detection of private key in dictionary"""
        unsafe_data = {
            "wallet_id": "test_wallet",
            "private_key": "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef"
        }
        
        assert WalletSecurityValidator.validate_no_private_key_exposure(unsafe_data) is False
    
    def test_detect_mnemonic_phrase(self):
        """Test detection of mnemonic phrase"""
        mnemonic = "abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about"
        
        assert WalletSecurityValidator.validate_no_private_key_exposure(mnemonic) is False
    
    def test_detect_sensitive_keys(self):
        """Test detection of sensitive dictionary keys"""
        unsafe_data = {
            "wallet_id": "test_wallet",
            "secret": "some_secret_value"
        }
        
        assert WalletSecurityValidator.validate_no_private_key_exposure(unsafe_data) is False
    
    def test_sanitize_for_llm(self):
        """Test data sanitization for LLM"""
        unsafe_data = {
            "wallet_id": "test_wallet",
            "address": "0x1234567890123456789012345678901234567890",
            "private_key": "secret_key",
            "balance": 1.5,
            "nested": {
                "mnemonic": "secret_phrase",
                "safe_field": "safe_value"
            }
        }
        
        sanitized = WalletSecurityValidator.sanitize_for_llm(unsafe_data)
        
        assert sanitized["wallet_id"] == "test_wallet"
        assert sanitized["address"] == "0x1234567890123456789012345678901234567890"
        assert sanitized["private_key"] == "[REDACTED]"
        assert sanitized["balance"] == 1.5
        assert sanitized["nested"]["mnemonic"] == "[REDACTED]"
        assert sanitized["nested"]["safe_field"] == "safe_value"
    
    def test_validate_list_data(self):
        """Test validation of list data"""
        safe_list = ["wallet1", "wallet2", "0x1234567890123456789012345678901234567890"]
        unsafe_list = ["wallet1", "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef"]
        
        assert WalletSecurityValidator.validate_no_private_key_exposure(safe_list) is True
        assert WalletSecurityValidator.validate_no_private_key_exposure(unsafe_list) is False


class TestWalletSigning:
    """Test cases for wallet transaction signing"""
    
    @pytest.fixture
    def wallet_manager_with_wallet(self):
        """Create wallet manager with a test wallet"""
        manager = SecureWalletManager("test_password")
        
        with patch('utils.secure_wallet.keyring') as mock_keyring:
            # Mock encrypted private key storage
            encrypted_key = manager.encryption_key.encrypt(
                "0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef".encode()
            )
            mock_keyring.get_password.return_value = encrypted_key.decode()
            
            # Create wallet
            wallet_info = manager.create_wallet("test_wallet", WalletType.LOCAL_ENCRYPTED, "ethereum")
            
            yield manager, wallet_info
    
    def test_sign_transaction_success(self, wallet_manager_with_wallet):
        """Test successful transaction signing"""
        manager, wallet_info = wallet_manager_with_wallet
        
        transaction_data = {
            "to": "0x742d35Cc6634C0532925a3b8D4C9db96C4b4d8b6",
            "value": 1000000000000000000,  # 1 ETH in wei
            "gas": 21000,
            "gasPrice": 20000000000,  # 20 Gwei
            "nonce": 0
        }
        
        with patch('utils.secure_wallet.Account') as mock_account:
            mock_signed_txn = Mock()
            mock_signed_txn.rawTransaction.hex.return_value = "0xsigned_transaction_hex"
            mock_account.from_key.return_value.sign_transaction.return_value = mock_signed_txn
            
            signed_txn_hex = manager.sign_transaction("test_wallet", transaction_data)
            
            assert signed_txn_hex == "0xsigned_transaction_hex"
            mock_account.from_key.assert_called_once()
    
    def test_sign_transaction_wallet_not_found(self, wallet_manager_with_wallet):
        """Test signing with non-existent wallet"""
        manager, _ = wallet_manager_with_wallet
        
        transaction_data = {"to": "0x742d35Cc6634C0532925a3b8D4C9db96C4b4d8b6"}
        
        with pytest.raises(ValueError, match="Wallet nonexistent not found"):
            manager.sign_transaction("nonexistent", transaction_data)


@pytest.mark.asyncio
class TestAsyncWalletOperations:
    """Test cases for async wallet operations"""
    
    async def test_async_wallet_creation(self):
        """Test async wallet creation workflow"""
        manager = SecureWalletManager("test_password")
        
        with patch('utils.secure_wallet.keyring'):
            # Simulate async wallet creation
            wallet_info = manager.create_wallet("async_wallet", WalletType.LOCAL_ENCRYPTED, "ethereum")
            
            assert wallet_info.wallet_id == "async_wallet"
            assert wallet_info.address.startswith("0x")


class TestWalletExceptions:
    """Test exception handling in wallet operations"""
    
    def test_private_key_exposure_exception(self):
        """Test PrivateKeyExposureException"""
        with pytest.raises(PrivateKeyExposureException):
            raise PrivateKeyExposureException("Private key detected", "test_source")
    
    def test_wallet_security_exception(self):
        """Test WalletSecurityException"""
        with pytest.raises(WalletSecurityException):
            raise WalletSecurityException("Security violation", "test_wallet")
    
    def test_encryption_exception(self):
        """Test EncryptionException"""
        with pytest.raises(EncryptionException):
            raise EncryptionException("Encryption failed", "encrypt")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
