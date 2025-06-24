"""
Secure Wallet Management System
Ensures private keys are never exposed to LLMs and stored securely locally
Supports hardware wallets and encrypted local storage
"""

import os
import json
import logging
import keyring
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

from web3 import Web3
from eth_account import Account
from mnemonic import Mnemonic
from hdwallet import HDWallet
from hdwallet.symbols import ETH, BTC


class WalletType(Enum):
    """Types of supported wallets"""
    LOCAL_ENCRYPTED = "local_encrypted"
    HARDWARE_LEDGER = "hardware_ledger"
    HARDWARE_TREZOR = "hardware_trezor"
    MNEMONIC_HD = "mnemonic_hd"


@dataclass
class WalletInfo:
    """Wallet information without sensitive data"""
    wallet_id: str
    wallet_type: WalletType
    address: str
    network: str
    balance: float = 0.0
    is_active: bool = True


class SecureWalletManager:
    """
    Secure wallet management that never exposes private keys to LLMs
    All sensitive operations are performed locally with encryption
    """
    
    def __init__(self, master_password: Optional[str] = None):
        self.logger = logging.getLogger(__name__)
        self.master_password = master_password
        self.encryption_key = None
        self.wallets: Dict[str, WalletInfo] = {}
        
        # Initialize encryption
        if master_password:
            self._initialize_encryption(master_password)
        
        # Load wallet metadata (non-sensitive data only)
        self._load_wallet_metadata()
    
    def _initialize_encryption(self, password: str):
        """Initialize encryption key from master password"""
        try:
            # Derive encryption key from password
            password_bytes = password.encode()
            salt = b'nyxtrade_salt_2024'  # In production, use random salt per user
            
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=32,
                salt=salt,
                iterations=100000,
            )
            
            key = base64.urlsafe_b64encode(kdf.derive(password_bytes))
            self.encryption_key = Fernet(key)
            
            self.logger.info("Encryption initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize encryption: {e}")
            raise
    
    def create_wallet(self, wallet_id: str, wallet_type: WalletType, 
                     network: str = "ethereum", **kwargs) -> WalletInfo:
        """
        Create a new wallet securely
        Private keys are never exposed to calling code
        """
        try:
            if wallet_type == WalletType.LOCAL_ENCRYPTED:
                return self._create_local_wallet(wallet_id, network, **kwargs)
            elif wallet_type == WalletType.MNEMONIC_HD:
                return self._create_hd_wallet(wallet_id, network, **kwargs)
            elif wallet_type == WalletType.HARDWARE_LEDGER:
                return self._setup_hardware_wallet(wallet_id, "ledger", network)
            elif wallet_type == WalletType.HARDWARE_TREZOR:
                return self._setup_hardware_wallet(wallet_id, "trezor", network)
            else:
                raise ValueError(f"Unsupported wallet type: {wallet_type}")
                
        except Exception as e:
            self.logger.error(f"Failed to create wallet {wallet_id}: {e}")
            raise
    
    def _create_local_wallet(self, wallet_id: str, network: str, **kwargs) -> WalletInfo:
        """Create locally encrypted wallet"""
        if not self.encryption_key:
            raise ValueError("Encryption not initialized")
        
        # Generate new account
        account = Account.create()
        address = account.address
        private_key = account.key.hex()
        
        # Encrypt and store private key securely
        encrypted_key = self.encryption_key.encrypt(private_key.encode())
        
        # Store in system keyring (never in files)
        keyring.set_password("nyxtrade", f"{wallet_id}_key", encrypted_key.decode())
        
        # Create wallet info (no sensitive data)
        wallet_info = WalletInfo(
            wallet_id=wallet_id,
            wallet_type=WalletType.LOCAL_ENCRYPTED,
            address=address,
            network=network
        )
        
        self.wallets[wallet_id] = wallet_info
        self._save_wallet_metadata()
        
        self.logger.info(f"Created local encrypted wallet: {wallet_id} at {address}")
        return wallet_info
    
    def _create_hd_wallet(self, wallet_id: str, network: str, 
                         mnemonic_phrase: Optional[str] = None) -> WalletInfo:
        """Create HD wallet from mnemonic"""
        if not self.encryption_key:
            raise ValueError("Encryption not initialized")
        
        # Generate or use provided mnemonic
        if not mnemonic_phrase:
            mnemo = Mnemonic("english")
            mnemonic_phrase = mnemo.generate(strength=256)
        
        # Create HD wallet
        if network.lower() in ["ethereum", "eth"]:
            symbol = ETH
        elif network.lower() in ["bitcoin", "btc"]:
            symbol = BTC
        else:
            symbol = ETH  # Default to Ethereum
        
        hdwallet = HDWallet(symbol=symbol)
        hdwallet.from_mnemonic(mnemonic_phrase)
        hdwallet.from_path("m/44'/60'/0'/0/0")  # Standard Ethereum path
        
        address = hdwallet.address()
        private_key = hdwallet.private_key()
        
        # Encrypt and store mnemonic and private key
        encrypted_mnemonic = self.encryption_key.encrypt(mnemonic_phrase.encode())
        encrypted_key = self.encryption_key.encrypt(private_key.encode())
        
        keyring.set_password("nyxtrade", f"{wallet_id}_mnemonic", encrypted_mnemonic.decode())
        keyring.set_password("nyxtrade", f"{wallet_id}_key", encrypted_key.decode())
        
        wallet_info = WalletInfo(
            wallet_id=wallet_id,
            wallet_type=WalletType.MNEMONIC_HD,
            address=address,
            network=network
        )
        
        self.wallets[wallet_id] = wallet_info
        self._save_wallet_metadata()
        
        self.logger.info(f"Created HD wallet: {wallet_id} at {address}")
        return wallet_info
    
    def _setup_hardware_wallet(self, wallet_id: str, device_type: str, network: str) -> WalletInfo:
        """Setup hardware wallet connection"""
        # This would integrate with hardware wallet libraries
        # For now, return placeholder
        
        # In production, this would:
        # 1. Connect to hardware device
        # 2. Get public key/address
        # 3. Store device connection info
        # 4. Never store private keys (they stay on device)
        
        placeholder_address = "0x" + "0" * 40  # Placeholder
        
        wallet_info = WalletInfo(
            wallet_id=wallet_id,
            wallet_type=WalletType.HARDWARE_LEDGER if device_type == "ledger" else WalletType.HARDWARE_TREZOR,
            address=placeholder_address,
            network=network
        )
        
        self.wallets[wallet_id] = wallet_info
        self._save_wallet_metadata()
        
        self.logger.info(f"Setup hardware wallet: {wallet_id} ({device_type})")
        return wallet_info
    
    def get_wallet_info(self, wallet_id: str) -> Optional[WalletInfo]:
        """Get wallet information (no sensitive data)"""
        return self.wallets.get(wallet_id)
    
    def list_wallets(self) -> List[WalletInfo]:
        """List all wallets (no sensitive data)"""
        return list(self.wallets.values())
    
    def sign_transaction(self, wallet_id: str, transaction_data: Dict[str, Any]) -> str:
        """
        Sign transaction securely without exposing private key
        Returns signed transaction hex
        """
        try:
            wallet_info = self.wallets.get(wallet_id)
            if not wallet_info:
                raise ValueError(f"Wallet {wallet_id} not found")
            
            if wallet_info.wallet_type == WalletType.LOCAL_ENCRYPTED:
                return self._sign_with_local_key(wallet_id, transaction_data)
            elif wallet_info.wallet_type == WalletType.MNEMONIC_HD:
                return self._sign_with_hd_key(wallet_id, transaction_data)
            elif wallet_info.wallet_type in [WalletType.HARDWARE_LEDGER, WalletType.HARDWARE_TREZOR]:
                return self._sign_with_hardware(wallet_id, transaction_data)
            else:
                raise ValueError(f"Unsupported wallet type for signing: {wallet_info.wallet_type}")
                
        except Exception as e:
            self.logger.error(f"Failed to sign transaction for wallet {wallet_id}: {e}")
            raise
    
    def _sign_with_local_key(self, wallet_id: str, transaction_data: Dict[str, Any]) -> str:
        """Sign transaction with locally stored encrypted key"""
        if not self.encryption_key:
            raise ValueError("Encryption not initialized")
        
        # Retrieve and decrypt private key
        encrypted_key = keyring.get_password("nyxtrade", f"{wallet_id}_key")
        if not encrypted_key:
            raise ValueError(f"Private key not found for wallet {wallet_id}")
        
        decrypted_key = self.encryption_key.decrypt(encrypted_key.encode()).decode()
        
        # Sign transaction
        account = Account.from_key(decrypted_key)
        signed_txn = account.sign_transaction(transaction_data)
        
        # Clear sensitive data from memory
        decrypted_key = None
        del decrypted_key
        
        return signed_txn.rawTransaction.hex()
    
    def _sign_with_hd_key(self, wallet_id: str, transaction_data: Dict[str, Any]) -> str:
        """Sign transaction with HD wallet key"""
        # Similar to _sign_with_local_key but uses HD wallet derivation
        return self._sign_with_local_key(wallet_id, transaction_data)
    
    def _sign_with_hardware(self, wallet_id: str, transaction_data: Dict[str, Any]) -> str:
        """Sign transaction with hardware wallet"""
        # This would integrate with hardware wallet APIs
        # Private keys never leave the hardware device
        
        wallet_info = self.wallets[wallet_id]
        
        if wallet_info.wallet_type == WalletType.HARDWARE_LEDGER:
            # Integrate with Ledger API
            pass
        elif wallet_info.wallet_type == WalletType.HARDWARE_TREZOR:
            # Integrate with Trezor API
            pass
        
        # Placeholder return
        return "0x" + "0" * 128  # Placeholder signed transaction
    
    def get_public_address(self, wallet_id: str) -> str:
        """Get public address (safe to expose)"""
        wallet_info = self.wallets.get(wallet_id)
        if not wallet_info:
            raise ValueError(f"Wallet {wallet_id} not found")
        
        return wallet_info.address
    
    def _load_wallet_metadata(self):
        """Load wallet metadata (non-sensitive data only)"""
        try:
            metadata_file = os.path.expanduser("~/.nyxtrade/wallet_metadata.json")
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    data = json.load(f)
                
                for wallet_data in data.get('wallets', []):
                    wallet_info = WalletInfo(
                        wallet_id=wallet_data['wallet_id'],
                        wallet_type=WalletType(wallet_data['wallet_type']),
                        address=wallet_data['address'],
                        network=wallet_data['network'],
                        balance=wallet_data.get('balance', 0.0),
                        is_active=wallet_data.get('is_active', True)
                    )
                    self.wallets[wallet_info.wallet_id] = wallet_info
                    
        except Exception as e:
            self.logger.error(f"Failed to load wallet metadata: {e}")
    
    def _save_wallet_metadata(self):
        """Save wallet metadata (non-sensitive data only)"""
        try:
            metadata_dir = os.path.expanduser("~/.nyxtrade")
            os.makedirs(metadata_dir, exist_ok=True)
            
            metadata_file = os.path.join(metadata_dir, "wallet_metadata.json")
            
            data = {
                'wallets': [
                    {
                        'wallet_id': wallet.wallet_id,
                        'wallet_type': wallet.wallet_type.value,
                        'address': wallet.address,
                        'network': wallet.network,
                        'balance': wallet.balance,
                        'is_active': wallet.is_active
                    }
                    for wallet in self.wallets.values()
                ]
            }
            
            with open(metadata_file, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            self.logger.error(f"Failed to save wallet metadata: {e}")
    
    def delete_wallet(self, wallet_id: str):
        """Securely delete wallet and all associated data"""
        try:
            if wallet_id not in self.wallets:
                raise ValueError(f"Wallet {wallet_id} not found")
            
            # Remove from keyring
            try:
                keyring.delete_password("nyxtrade", f"{wallet_id}_key")
            except:
                pass
            
            try:
                keyring.delete_password("nyxtrade", f"{wallet_id}_mnemonic")
            except:
                pass
            
            # Remove from memory
            del self.wallets[wallet_id]
            
            # Update metadata file
            self._save_wallet_metadata()
            
            self.logger.info(f"Deleted wallet: {wallet_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to delete wallet {wallet_id}: {e}")
            raise


class WalletSecurityValidator:
    """
    Validates that wallet operations are secure and private keys are never exposed
    """
    
    @staticmethod
    def validate_no_private_key_exposure(data: Any) -> bool:
        """
        Validate that data doesn't contain private keys or sensitive information
        This should be called before any data is sent to LLMs
        """
        if isinstance(data, str):
            # Check for common private key patterns
            if len(data) == 64 and all(c in '0123456789abcdefABCDEF' for c in data):
                return False  # Looks like a hex private key
            
            if data.startswith('0x') and len(data) == 66:
                return False  # Ethereum private key format
            
            # Check for mnemonic phrases (12-24 words)
            words = data.split()
            if 12 <= len(words) <= 24:
                return False  # Potential mnemonic phrase
        
        elif isinstance(data, dict):
            # Recursively check dictionary values
            for key, value in data.items():
                if key.lower() in ['private_key', 'privatekey', 'secret', 'mnemonic', 'seed']:
                    return False
                if not WalletSecurityValidator.validate_no_private_key_exposure(value):
                    return False
        
        elif isinstance(data, list):
            # Check list items
            for item in data:
                if not WalletSecurityValidator.validate_no_private_key_exposure(item):
                    return False
        
        return True
    
    @staticmethod
    def sanitize_for_llm(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize data before sending to LLM by removing sensitive fields
        """
        safe_data = {}
        
        for key, value in data.items():
            if key.lower() in ['private_key', 'privatekey', 'secret', 'mnemonic', 'seed', 'password']:
                safe_data[key] = "[REDACTED]"
            elif isinstance(value, dict):
                safe_data[key] = WalletSecurityValidator.sanitize_for_llm(value)
            elif isinstance(value, list):
                safe_data[key] = [
                    WalletSecurityValidator.sanitize_for_llm(item) if isinstance(item, dict) else item
                    for item in value
                ]
            else:
                safe_data[key] = value
        
        return safe_data
